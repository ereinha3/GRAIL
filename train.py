import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx

from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PrefixTuningConfig, PeftModelForCausalLM

from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from model import GraphSAGEEncoder, Projector
from const import MODELS_PATH, COMPATIBLE_MODELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.set_verbosity_error()

def train():
    #
    # 0) Either generate or load an existing synthetic TSP dataset.
    #
    task = TSP_Task()

    # If you have not already created the pickles, uncomment:
    #    task.generate_dataset()
    # Otherwise, just load from disk:
    try:
        task.load_dataset()
    except FileNotFoundError:
        task.generate_dataset()
        task.load_dataset()

    for split in ['easy']:

        train_probs = task.problem_set[split]['train']
        val_probs   = task.problem_set[split]['val']

        train_dataset = TSPDataset(train_probs)
        val_dataset   = TSPDataset(val_probs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )

        max_deg = 0
        for prob in train_probs:
            g = prob['graph']
            degs = [d for _, d in g.degree()]
            max_deg = max(max_deg, max(degs))

        H = 128   # hidden dim for GraphSAGE
        K = 8     # number of clusters (METIS blocks / prefix tokens)

        for model_code in COMPATIBLE_MODELS.keys():
            
            model_name = COMPATIBLE_MODELS[model_code]

            encoder = GraphSAGEEncoder(
                max_degree=max_deg,
                node_embed_dim=32,
                hidden_dim=H,
                num_layers=2,
                num_clusters=K
            ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
            ).to(device)

            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()


            base_model.config.pad_token_id = tokenizer.eos_token_id

            projector = Projector(H, base_model.config.hidden_size).to(device)

            optimizer = torch.optim.Adam(
                list(encoder.parameters()) +
                list(projector.parameters()),
                lr=1e-3
            )

            best_val_acc = 0
            patience      = 20
            validate_every = 5
            no_improve    = 0
            epoch_counter = 0

            if split == 'easy':
                max_length = 512
            elif split == 'hard':
                max_length = 2048

            while True:
                train_loss = 0.0
                encoder.train()

                for batch_data, prompts, solutions in train_loader:

                    batch_data = batch_data.to(device)

                    # GNN → cluster vectors [B, K, H]
                    cluster_vectors = encoder(batch_data)
                    prefix_embeds   = projector(cluster_vectors).to(base_model.device)  # [B, K, D_model]

                    # Build the prompt + solution string for each example in the batch.
                    #     We assume `solutions` is a list of lists of airport‐codes, so solutions[i] is e.g. ["A3","A1",...].
                    full_texts = []
                    for i, sol_codes in enumerate(solutions):
                        sol_str = "[" + ", ".join(sol_codes) + "]"
                        full_texts.append(prompts[i] + "\nSolution: " + sol_str)

                    # (c) Tokenize just the prompt (to learn prompt_len)
                    enc_prompts = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    prompt_len = enc_prompts.input_ids.shape[1]

                    # (d) Tokenize the full text (prompt + solution)
                    enc_full = tokenizer(
                        full_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    full_ids = enc_full.input_ids            # [B, L_total]
                    att_mask = enc_full.attention_mask       # [B, L_total]

                    # Convert input_ids to embeddings
                    input_embeds = base_model.get_input_embeddings()(full_ids)

                    full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

                    # Assume: full_ids.shape = [B, L] and prefix_embeds.shape = [B, K, D]
                    K = prefix_embeds.shape[1]

                    # Mask all prefix positions
                    prefix_labels = torch.full((full_ids.size(0), K), -100, dtype=full_ids.dtype).to(full_ids.device)
                    labels = torch.cat([prefix_labels, full_ids], dim=1)

                    # Mask prompt tokens if needed (already handled before)
                    labels[:, K:K+prompt_len] = -100  # Keep solution tokens only

                    prefix_mask = torch.ones((att_mask.size(0), K), dtype=torch.long).to(att_mask.device)
                    att_mask = torch.cat([prefix_mask, att_mask], dim=1)

                    # (f) Forward through base_model
                    outputs = base_model(
                        inputs_embeds   = full_embeds,
                        attention_mask  = att_mask,
                        labels          = labels
                    )

                    loss = outputs.loss
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss /= len(train_loader)
                print(f"[Epoch {epoch_counter}] Train Loss: {train_loss:.4f}")

                if epoch_counter % validate_every == 0:
                    encoder.eval()
                    val_acc = 0.0

                    with torch.no_grad():
                        for batch_data, prompts, solutions in val_loader:
                            batch_data = batch_data.to(device)

                            # (a) GNN → prefix
                            cluster_vectors = encoder(batch_data)
                            prefix_embeds = projector(cluster_vectors).to(base_model.device)

                            # (b) Build full_texts again
                            prompts_with_cue = [p + "\nSolution:" for p in prompts]

                            toks = tokenizer(
                                prompts_with_cue, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=max_length
                                ).to(device)
                            input_ids = toks.input_ids.to(device)
                            prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, 768]
                            att_mask = toks.attention_mask.to(device)

                            full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

                            prefix_mask = torch.ones((att_mask.size(0), K), dtype=torch.long).to(att_mask.device)
                            att_mask = torch.cat([prefix_mask, att_mask], dim=1)
                            
                            out = base_model.generate(
                                inputs_embeds = full_embeds,
                                attention_mask = att_mask,
                                max_new_tokens = 60,
                                do_sample = False,
                                pad_token_id = tokenizer.pad_token_id,
                                eos_token_id = tokenizer.eos_token_id
                            )

                            batch_val_acc = 0

                            for i in range(len(out)):
                                raw_text = tokenizer.decode(out[i], skip_special_tokens=True)
                                if "]" in raw_text:
                                    truncated = raw_text[: raw_text.find("]") + 1]
                                    truncated = truncated.strip()
                                    truncated = truncated.replace("[", "").replace("]", "")
                                    truncated = truncated.split(", ")
                                    if len(truncated) == len(solutions[i]):
                                        batch_val_acc += 1 if sum(1 for a, b in zip(truncated, solutions[i]) if a == b) == len(truncated) else 0

                            batch_val_acc /= len(out)
                            val_acc += batch_val_acc

                    val_acc /= len(val_loader)
                    print(f"[Epoch {epoch_counter}] Val Acc: {val_acc:.4f}")

                    # 6) Early‐stopping check
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        if split == 'easy':
                            torch.save(encoder.state_dict(), MODELS_PATH + 'easy_encode_' + model_code + '.pth')
                            torch.save(projector.state_dict(), MODELS_PATH + 'easy_projector_' + model_code + '.pth')
                        elif split == 'hard':
                            torch.save(encoder.state_dict(), MODELS_PATH + 'hard_encode_' + model_code + '.pth')
                            torch.save(projector.state_dict(), MODELS_PATH + 'hard_projector_' + model_code + '.pth')
                        no_improve = 0
                        print("→ New best validation loss; saved checkpoints.")
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print("Early stopping triggered.")
                            break

                if no_improve >= patience:
                    break

                epoch_counter += 1


            print(f"Training complete for {model_name}.")
            # Free GPU memory explicitly
            del encoder, projector, base_model, tokenizer, optimizer
            torch.cuda.empty_cache()


    print("Training complete.")

if __name__ == "__main__":
    train()
