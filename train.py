import torch
from torch.utils.data import DataLoader
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import argparse
import torch.nn as nn
from torch.amp import GradScaler, autocast
import math
import random
import numpy as np
from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from const import MODELS_PATH, COMPATIBLE_MODELS, H, K, MAX_INPUT_LENGTH, B, ENCODERS
from utils import add_solution_to_prompt, add_solution_cue_to_prompt, StopOnBracket, print_model_input, get_global_max_prompt_length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.set_verbosity_error()

def train(model_code, encoder_code):

    task = TSP_Task()

    try:
        task.load_dataset()
    except FileNotFoundError:
        task.generate_dataset()
        task.load_dataset()


    # train_probs = task.problem_set['easy']['train']
    # val_probs   = task.problem_set['easy']['val']

    train_probs = task.problem_set['easy']['train'] + task.problem_set['hard']['train']
    val_probs   = task.problem_set['easy']['val'] + task.problem_set['hard']['val']

    longest_input = MAX_INPUT_LENGTH

    train_dataset = TSPDataset(train_probs)
    val_dataset   = TSPDataset(val_probs)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(1234)

    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    b = math.ceil(B / 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=b,
        shuffle=False,
        collate_fn=collate_fn
    )

    max_deg = task.hard_max_nodes - 1
        
    model_name = COMPATIBLE_MODELS[model_code]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
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

    print(llm_dim)
    encoder = ENCODERS[encoder_code](
        max_degree=max_deg,
        node_embed_dim=32,
        hidden_dim=H,
        num_layers=3,
        num_clusters=K,
        output_dim=llm_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()),
        lr=1e-4
    )

    best_val_acc    = float('-inf')
    patience        = 10
    validate_every  = 5
    no_improve      = 0
    epoch_counter   = 0

    while True:

        train_loss = 0.0
        encoder.train()

        for batch_data, prompts, solutions in train_loader:

            optimizer.zero_grad()


            batch_data = batch_data.to(device)

            # GNN -> cluster vectors [B, K, H]
            prefix_embeds = encoder(batch_data).to(device)

            # Build the prompt + solution string for each example in the batch. Essential for 'teacher-forcing'
            full_texts = [add_solution_to_prompt(p, s) for p, s in zip(prompts, solutions)]

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

            # This will print the input to the model to assure labels are properly aligned and embeddings look good
            # for i in range(B):
            #     if len(solutions[i]) == (task.hard_max_nodes+1):
            #         print_model_input(full_texts, enc_prompts, enc_full, labels, prompt_len, tokenizer)
            #         break
            

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
            
        train_loss /= len(train_loader)

        print(f"[Epoch {epoch_counter}] Train Loss: {train_loss:.4f}")

        if epoch_counter % validate_every == 0:
            encoder.eval()
            val_acc = 0.0

            with torch.no_grad():
                for batch_data, prompts, solutions in val_loader:
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
                    input_ids = toks.input_ids.to(device)
                    prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, LLM_DIM]
                    att_mask = toks.attention_mask.to(device)

                    full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

                    prefix_mask = torch.ones((b, K), dtype=torch.long).to(device)
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

                    for i in range(b):
                        raw_text = tokenizer.decode(out[i], skip_special_tokens=True)
                        truncated = raw_text.strip().replace("[", "").replace("]", "").split(", ")
                        if truncated == solutions[i] or truncated == solutions[i][::-1]:
                            batch_val_acc += 1

                    batch_val_acc /= b
                    val_acc += batch_val_acc

            val_acc /= len(val_loader)
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






if __name__ == "__main__":
    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(
        description="Train a model for TSP (Traveling Salesman Problem) using various encoders and language models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_code",
        type=str,
        default="opt125",
        choices=list(COMPATIBLE_MODELS.keys()),
        help="Code for the language model to use. Available options: " + ", ".join(COMPATIBLE_MODELS.keys())
    )
    
    parser.add_argument(
        "--encoder",
        type=str,
        default="SAGEEncoder",
        choices=list(ENCODERS.keys()),
        help="Code for the encoder architecture to use. Available options: " + ", ".join(ENCODERS.keys())
    )

    
    args = parser.parse_args()

    print(f"Starting training with:")
    print(f"  Model: {args.model_code}")
    print(f"  Encoder: {args.encoder}")
    print("-" * 50)
    
    train(args.model_code, args.encoder)
