import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, PeftModelForCausalLM

from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from model import GraphSAGEEncoder, Projector
from collections import defaultdict
from const import MODELS_PATH, COMPATIBLE_MODELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference():
    #
    # 0) Load the same synthetic TSP dataset that training used.
    #
    task = TSP_Task()
    task.load_dataset()
    test_probs = task.problem_set["easy"]["test"]

    #
    # 1) Wrap the test split in TSPDataset → DataLoader
    #
    B = 1  # we’ll run one example at a time
    test_dataset = TSPDataset(test_probs)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=False,
        collate_fn=collate_fn
    )

    #
    # 2) Reconstruct GraphSAGEEncoder and Projector exactly as in train.py
    #
    max_deg = 0
    for prob in task.problem_set["easy"]["train"]:
        g = prob["graph"]
        max_deg = max(max_deg, max(d for _, d in g.degree()))

    H = 128   # GraphSAGE hidden dimension
    K = 8     # number of “virtual prefix tokens”

    encoder = GraphSAGEEncoder(
        max_degree     = max_deg,
        node_embed_dim = 32,
        hidden_dim     = H,
        num_layers     = 2,
        num_clusters   = K
    ).to(device)

    for model_code in COMPATIBLE_MODELS.keys():

        model_name = COMPATIBLE_MODELS[model_code]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(device)

        # Set pad token to eos token
        base_model.config.pad_token_id = tokenizer.eos_token_id

        projector = Projector(H, base_model.config.hidden_size).to(device)

        encoder.load_state_dict(torch.load(
            MODELS_PATH + "easy_encode_"    + model_code + ".pth",
            map_location=device
        ))
        projector.load_state_dict(torch.load(
            MODELS_PATH + "easy_projector_" + model_code + ".pth",
            map_location=device
        ))

        encoder.eval()
        projector.eval()

        size_counts = defaultdict(int)
        metrics_by_size = defaultdict(lambda: {"acc": 0, "feas": 0, "hall": 0})

        for batch_data, prompts, solutions in test_loader:

            batch_data = batch_data.to(device)

            with torch.no_grad():
                # Compute prefix
                graph_features = encoder(batch_data)           # [B, K, 128]
                projected = projector(graph_features)           # [B, K, 768]
                            
                prompts_with_cue = [p + "\nSolution:" for p in prompts]  # length = B = 1

                # Tokenize prompt
                toks = tokenizer(prompts_with_cue, return_tensors="pt", padding=False)
                input_ids = toks.input_ids.to(device)
                prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, 768]

                # Prepend soft prefix
                full_embeds = torch.cat([projected, prompt_embeds], dim=1)    # [B, K+M, 768]
                full_mask   = torch.cat([torch.ones(projected.shape[:2], dtype=torch.long, device=device),
                                        toks.attention_mask.to(device)], dim=1)

                # Call generate on base model directly
                out = base_model.generate(
                    inputs_embeds = full_embeds,
                    attention_mask = full_mask,
                    max_new_tokens = 30,
                    do_sample = False,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id
                )

            # Decode generated ids to final string
            for i in range(len(out)):
                raw_text = tokenizer.decode(out[i], skip_special_tokens=True)
                if "]" in raw_text:
                    truncated = raw_text[: raw_text.find("]") + 1]
                    truncated = truncated.strip()
                    truncated = truncated.replace("[", "").replace("]", "")
                    truncated = truncated.split(", ")

                    num_nodes = len(solutions[i]) - 1
                    size_counts[num_nodes] += 1

                    duplicate_nodes = sum([truncated.count(ele) - 1 for ele in set(truncated)])

                    if sum(1 for token in truncated if token not in solutions[i]) > 0:
                        metrics_by_size[num_nodes]["hall"] += 1
                    elif len(truncated) != len(solutions[i]):
                        metrics_by_size[num_nodes]["hall"] += 1
                    elif duplicate_nodes > 1:
                        metrics_by_size[num_nodes]["hall"] += 1
                    else:
                        if truncated[-1] == truncated[0]:
                            metrics_by_size[num_nodes]["feas"] += 1
                        
                        if truncated == solutions[i] or truncated[::-1] == solutions[i]:
                            metrics_by_size[num_nodes]["acc"] += 1

                    print('-' * 100)
                    if sum(1 for token in truncated if token not in solutions[i]) > 0:
                        print('Hallucinating: made up a node.')
                    elif len(truncated) == len(solutions[i]):
                        if truncated == solutions[i]:
                            print('Correct.')
                        elif truncated[::-1] == solutions[i]:
                            print('Correct, but reversed.')
                        else:
                            print('Not hallucinating, but wrong.')
                    else:
                        print('Hallucinating: wrong lengths.')
                    print('GT:  ', solutions[i])
                    print('PRED:', truncated)
                    print('-' * 100)
            print(metrics_by_size)
        
        for size in sorted(size_counts.keys()):
            print('-' * 100)
            print(f"Size {size}: {size_counts[size]}")
            print(f"Accuracy: {metrics_by_size[size]['acc'] / size_counts[size]:.4f}")
            print(f"Feasibility: {metrics_by_size[size]['feas'] / size_counts[size]:.4f}")
            print(f"Hallucination: {metrics_by_size[size]['hall'] / size_counts[size]:.4f}")
            print('-' * 100)

if __name__ == "__main__":
    inference()
