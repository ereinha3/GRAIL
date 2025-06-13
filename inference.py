import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import json
import os
from tqdm import tqdm

from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from collections import defaultdict
from const import MODELS_PATH, COMPATIBLE_MODELS, H, K, ENCODERS, OUTPUT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model_code, encoder_code, verbose):
    # Load tsp dataset
    task = TSP_Task()
    task.load_dataset()
    test_probs = task.problem_set["easy"]["test"] + task.problem_set["hard"]["test"]


    B = 1  # we’ll run one example at a time
    test_dataset = TSPDataset(test_probs)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=False,
        collate_fn=collate_fn
    )

    max_deg = task.hard_max_nodes - 1


    model_name = COMPATIBLE_MODELS[model_code]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    class StopOnBracket(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            # if last token decodes to ']', stop
            return tokenizer.decode([input_ids[0, -1].item()]).strip() == "]"

    stop_crit = StoppingCriteriaList([StopOnBracket()])

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
    ).to(device)

    # Set pad token to eos token
    base_model.config.pad_token_id = tokenizer.eos_token_id

    cfg = base_model.config
    if hasattr(cfg, "hidden_size"):
        llm_dim = cfg.hidden_size
    else:
        llm_dim = cfg.n_embd

    if encoder_code == 'None':
        encoder = None
    else:
        encoder = ENCODERS[encoder_code](
            max_degree=max_deg,
            node_embed_dim=32,
            hidden_dim=H,
            num_layers=3,
            num_clusters=K,
            output_dim=llm_dim
        ).to(device)


        encoder.load_state_dict(torch.load(
            MODELS_PATH + 'encoder_' + model_code + '_' + encoder_code + '.pth',
            map_location=device
        ))

        encoder.eval()

    size_counts = defaultdict(int)

    def fresh_hallucination_dict():
        return {
            "wrong_format": 0,
            "wrong_length": 0,
            "duplicate_nodes": 0,
            "made_up_node": 0,
        }

    metrics_by_size = defaultdict(lambda: {"acc": 0, "feas": 0, "hall": fresh_hallucination_dict()})

    for batch_data, prompts, solutions in tqdm(test_loader, desc="Processing test set"):

        batch_data = batch_data.to(device)

        with torch.no_grad():
                        
            prompts_with_cue = [p + "\nSolution:" for p in prompts] 

            # Tokenize prompt
            toks = tokenizer(prompts_with_cue, return_tensors="pt", padding=False)
            prompt_mask = toks.attention_mask.to(device)
            input_ids = toks.input_ids.to(device)

            if encoder_code == 'None':
                # Call generate on base model directly
                out = base_model.generate(
                input_ids = input_ids,
                attention_mask = prompt_mask,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                stopping_criteria = stop_crit,
            )
            
            else:
                prefix_embeds = encoder(batch_data)           # [B, K, 128]
                prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, 768]

                # Prepend soft prefix
                full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)    # [B, K+M, 768]
                full_mask   = torch.cat([torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device),
                                        prompt_mask], dim=1)

            
                # Generate with prefix
                out = base_model.generate(
                    inputs_embeds = full_embeds,
                    attention_mask = full_mask,
                    do_sample = False,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    stopping_criteria = stop_crit,
                    max_new_tokens = 64
                )

        for i in range(len(solutions)):
            num_nodes = len(solutions[i]) - 1
            size_counts[num_nodes] += 1
            # Decode generated ids to final string
            
            
        for embeds in out:
            raw_text = tokenizer.decode(embeds, skip_special_tokens=True)
            if verbose:
                print(raw_text)
            if "]" in raw_text:
                truncated = raw_text[: raw_text.find("]") + 1]
                truncated = truncated.strip()
                truncated = truncated.replace("[", "").replace("]", "")
                truncated = truncated.split(", ")

                duplicate_nodes = sum([truncated.count(ele) - 1 for ele in set(truncated)])

                
                if len(truncated) != len(solutions[i]):
                    metrics_by_size[num_nodes]["hall"]["wrong_length"] += 1
                    if verbose:
                        print(f"Hallucinating: wrong lengths.")
                elif sum(1 for token in truncated if token not in solutions[i]) > 0:
                    metrics_by_size[num_nodes]["hall"]["made_up_node"] += 1
                    if verbose:
                        print(f"Hallucinating: made up a node.")
                elif duplicate_nodes > 1:
                    metrics_by_size[num_nodes]["hall"]["duplicate_nodes"] += 1
                    if verbose:
                        print(f"Hallucinating: duplicate nodes.")
                else:
                    if truncated == solutions[i] or truncated[::-1] == solutions[i]:
                        metrics_by_size[num_nodes]["acc"] += 1
                        if verbose:
                            print(f"Correct: solution found.")
                    elif truncated[-1] == truncated[0]:
                        metrics_by_size[num_nodes]["feas"] += 1
                        if verbose:
                            print(f"Feasible: last node == first node.")
                    else:
                        metrics_by_size[num_nodes]["hall"]["wrong_format"] += 1
                        if verbose:
                            print(f"Hallucinating: for some other reason.")

                if verbose:
                    print('GT:  ', solutions[i])
                    print('PRED:', truncated)
                
            else:
                metrics_by_size[num_nodes]["hall"]["wrong_format"] += 1
                if verbose:
                    print(f"Hallucinating: wrong format.")
                    print('GT:  ', solutions[i])
                    print('PRED:', raw_text)
            if verbose:
                print()

    json_path = OUTPUT_PATH + "accuracy.json"
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        with open(json_path, "w") as f:
            json.dump({}, f)

    with open(json_path, "r") as f:
        all_metrics = json.load(f)

    all_metrics.setdefault(model_code, {})
    all_metrics[model_code].setdefault(encoder_code, {})
    for size in metrics_by_size.keys():
        all_metrics[model_code][encoder_code].setdefault(size, {})
        all_metrics[model_code][encoder_code][size].setdefault('hall', {})

    for size in metrics_by_size.keys():
        for metric in metrics_by_size[size].keys():
            if metric != 'hall':
                all_metrics[model_code][encoder_code][size][metric] = metrics_by_size[size][metric] / size_counts[size]
            else:
                hall_count = sum([metrics_by_size[size]['hall'][hall_metric] for hall_metric in metrics_by_size[size]['hall'].keys()])
                for hall_metric in metrics_by_size[size]['hall'].keys():
                    if hall_count > 0:
                        all_metrics[model_code][encoder_code][size]['hall'][hall_metric] = metrics_by_size[size]['hall'][hall_metric] / hall_count
                    else:
                        all_metrics[model_code][encoder_code][size]['hall'][hall_metric] = 0
    
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


    if verbose:
        print('-' * 100)
        print(f"Model: {model_name} {'base' if encoder_code == 'None' else 'with prefix'}")
        for size in sorted(size_counts.keys()):
            print(f"Size {size}: {size_counts[size]}")
        print(f"Accuracy: {metrics_by_size[size]['acc'] / size_counts[size]:.4f}")
        print(f"Feasibility: {metrics_by_size[size]['feas'] / size_counts[size]:.4f}")
        print(f"Hallucination: {sum(val for val in metrics_by_size[size]['hall'].values()) / size_counts[size]:.4f}")
        for metric in metrics_by_size[size]['hall']:
            print(f"\t{metric}: {metrics_by_size[size]['hall'][metric] / size_counts[size]:.4f}")
        print('-' * 100)
        print()

if __name__ == "__main__":
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

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to print verbose output."
    )

    
    args = parser.parse_args()

    print(f"Starting training with:")
    print(f"  Model: {args.model_code}")
    print(f"  Encoder: {args.encoder}")
    print("-" * 50)
    
    inference(args.model_code, args.encoder, args.verbose)
