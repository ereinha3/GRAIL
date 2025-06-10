import torch
from torch.utils.data import DataLoader
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import argparse
import torch.nn as nn
from torch.amp import GradScaler, autocast

from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from model import SAGEEncoder, GINEncoder, GATEncoder, StopOnBracket
from const import MODELS_PATH, COMPATIBLE_MODELS, H, K, MAX_INPUT_LENGTH, B
from utils import get_max_prompt_length, add_solution_to_prompt, add_solution_cue_to_prompt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.set_verbosity_error()

def train(model_code):
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


    train_probs = task.problem_set['easy']['train'] + task.problem_set['hard']['train']
    val_probs   = task.problem_set['easy']['val'] + task.problem_set['hard']['val']
    # test_probs  = task.problem_set['easy']['test'] + task.problem_set['hard']['test']

    longest_input = MAX_INPUT_LENGTH

    # all_probs = train_probs + val_probs + test_probs

    
    # for model_code in COMPATIBLE_MODELS.keys():
    #     tokenizer = AutoTokenizer.from_pretrained(COMPATIBLE_MODELS[model_code])
    #     model = AutoModelForCausalLM.from_pretrained(COMPATIBLE_MODELS[model_code])

    #     max_length = get_max_prompt_length(all_probs, tokenizer, 8)
    #     max_tokens = tokenizer.model_max_length
    #     max_input = model.config.max_position_embeddings

    #     print(f"Max prompt length for {model_code}: {max_length}")
    #     print(f"Max input tokens for {model_code} tokenizer:", max_tokens)
    #     print(f"Positional embedding size for {model_code} input:", max_input)

    #     if max_length > max_tokens or max_length > max_input:
    #         raise ValueError(f"Max input tokens for {model_code} is too small.")
    #     longest_input = max(longest_input, max_length)

    #     del model, tokenizer
    #     torch.cuda.empty_cache()

    # print(f"Max input tokens: {longest_input}")

    train_dataset = TSPDataset(train_probs)
    val_dataset   = TSPDataset(val_probs)


    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=B,
        shuffle=False,
        collate_fn=collate_fn
    )

    max_deg = 0
    for prob in train_probs:
        g = prob['graph']
        degs = [d for _, d in g.degree()]
        max_deg = max(max_deg, max(degs))
        

        
    model_name = COMPATIBLE_MODELS[model_code]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto'
    )

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    base_model.config.pad_token_id = tokenizer.eos_token_id

    cfg = base_model.config
    if hasattr(cfg, "hidden_size"):
        llm_dim = cfg.hidden_size
    else:
        llm_dim = cfg.n_embd

    for encoder in [SAGEEncoder, GINEncoder, GATEncoder]:

        encoder = encoder(
            max_degree=max_deg,
            node_embed_dim=32,
            hidden_dim=H,
            num_layers=3,
            num_clusters=K,
            output_dim=llm_dim
        ).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()),
            lr=1e-3
        )

        best_val_acc    = 0
        patience        = 20
        validate_every  = 5
        no_improve      = 0
        epoch_counter   = 0

        while True:


            encoder, train_loss = train_epoch(encoder, tokenizer, base_model, train_loader, optimizer, longest_input)

            print(f"[Epoch {epoch_counter}] Train Loss: {train_loss:.4f}")

            if epoch_counter % validate_every == 0:
                val_acc = evaluate(encoder, tokenizer, base_model, val_loader, longest_input)
                print(f"[Epoch {epoch_counter}] Val Acc: {val_acc:.4f}")

                # 6) Early‐stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(encoder.state_dict(), MODELS_PATH + 'encoder_' + model_code + '_' + str(encoder) + '.pth')
                    no_improve = 0
                    print("→ New best validation loss; saved checkpoints.")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("Early stopping triggered.")
                        break

            epoch_counter += 1

        print(f"Training complete for {model_name} using {str(encoder)}.")

    print("Training complete.")

def train_epoch(encoder, tokenizer, base_model, loader, optimizer, longest_input):
    train_loss = 0.0
    encoder.train()

    for batch_data, prompts, solutions in loader:

        optimizer.zero_grad()


        batch_data = batch_data.to(device)

        # GNN → cluster vectors [B, K, H]
        prefix_embeds = encoder(batch_data).to(device)

        # Build the prompt + solution string for each example in the batch.
        #     We assume `solutions` is a list of lists of airport‐codes, so solutions[i] is e.g. ["A3","A1",...].
        full_texts = []
        for prompt, solution in zip(prompts, solutions):
            prompt_with_solution = add_solution_to_prompt(prompt, solution)
            full_texts.append(prompt_with_solution)

        # (c) Tokenize just the prompt (to learn prompt_len)
        enc_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=longest_input
        )

        prompt_len = enc_prompts.input_ids.shape[1]

        # (d) Tokenize the full text (prompt + solution)
        enc_full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=longest_input
        )
        full_ids = enc_full.input_ids.to(device)            # [B, L_total]
        att_mask = enc_full.attention_mask.to(device)       # [B, L_total]



        # Convert input_ids to embeddings
        input_embeds = base_model.get_input_embeddings()(full_ids)

        full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

        # Mask all prefix positions
        prefix_labels = torch.full((B, K), -100).to(device)
        labels = torch.cat([prefix_labels, full_ids], dim=1)

        labels[:, K:K+prompt_len] = -100

        # for i in range(B):
        #     print('='*100)
        #     print('Full prompt')
        #     print('\t', full_texts[i])
        #     print()
        #     print('Encoded prompts:')
        #     print('\t', enc_prompts.input_ids[i, :30])
        #     print('\t', tokenizer.decode(enc_prompts.input_ids[i, :30]))
        #     print('\t', enc_prompts.input_ids[i, -30:])
        #     print('\t', tokenizer.decode(enc_prompts.input_ids[i, -30:]))
        #     print()
        #     print('Encoded full:')
        #     print('\t', enc_full.input_ids[i, :30])
        #     print('\t', tokenizer.decode(enc_full.input_ids[i, :30]))
        #     print('\t', enc_full.input_ids[i, -30:])
        #     print('\t', tokenizer.decode(enc_full.input_ids[i, -30:]))
        #     print()
        #     print('Labels:')
        #     print('\t', labels[i, :30])
        #     print('\t These should all be -100')
        #     print('\t', labels[i, K + prompt_len:])
        #     print('\t', tokenizer.decode(labels[i,  K + prompt_len:]))
        #     print('='*100)
        
        if torch.isnan(full_embeds).any():
            print("NaN in full_embeds")
        if torch.isnan(labels).any():
            print("NaN in labels")
        if B != full_ids.shape[0]:
            print(B)
            print(full_ids.shape[0])
            exit()

        prefix_mask = torch.ones((B, K), dtype=torch.long).to(device)
        att_mask = torch.cat([prefix_mask, att_mask], dim=1)

        # (f) Forward through base_model

        outputs = base_model(
            inputs_embeds=full_embeds,
            attention_mask=att_mask,
            labels=labels,
        )
        loss = outputs.loss

        if torch.isnan(loss):
            print("NaN detected in loss. Debugging...")
            print("Input Embeds:", input_embeds)
            print("Prefix Embeds:", prefix_embeds)
            print("Labels:", labels)
            exit(1)

        batch_loss = loss.item() / B
        train_loss += batch_loss

        loss.backward()
        optimizer.step()
        
    return encoder, train_loss / len(loader)

def evaluate(encoder, tokenizer, base_model, loader, longest_input):

    encoder.eval()
    val_acc = 0.0

    with torch.no_grad():
        for batch_data, prompts, solutions in loader:
            batch_data = batch_data.to(device)

            # (a) GNN → prefix
            prefix_embeds = encoder(batch_data).to(device)

            # (b) Build full_texts again
            prompts_with_cue = [add_solution_cue_to_prompt(p) for p in prompts]

            toks = tokenizer(
                prompts_with_cue, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=longest_input
                )
            input_ids = toks.input_ids
            prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, 768]
            att_mask = toks.attention_mask

            full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

            prefix_mask = torch.ones((B, K), dtype=torch.long)
            att_mask = torch.cat([prefix_mask, att_mask], dim=1)
            
            out = base_model.generate(
                inputs_embeds = full_embeds,
                attention_mask = att_mask,
                max_new_tokens = 30,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                stopping_criteria = [StopOnBracket(tokenizer=tokenizer)]
            )

            batch_val_acc = 0

            for i in range(B):
                raw_text = tokenizer.decode(out[i], skip_special_tokens=True)
                truncated = raw_text.strip().replace("[", "").replace("]", "").split(", ")
                if truncated == solutions[i]:
                    batch_val_acc += 1

            batch_val_acc /= B
            val_acc += batch_val_acc

    val_acc /= len(loader)
    return val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_code", type=str, default="opt125")
    args = parser.parse_args()
    train(args.model_code)
