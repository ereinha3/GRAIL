import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from TSP import TSP_Task
from dataset import TSPDataset, collate_fn
from model import GraphSAGEEncoder
from collections import defaultdict
from const import MODELS_PATH, COMPATIBLE_MODELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def inference():
    #
    # 0) Load the same synthetic TSP dataset that training used.
    #
    task = TSP_Task()
    task.load_dataset()
    test_probs = task.problem_set["easy"]["test"] + task.problem_set["hard"]["test"]
    train_probs = task.problem_set["easy"]["train"] + task.problem_set["hard"]["train"]

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
    for prob in train_probs:
        g = prob["graph"]
        max_deg = max(max_deg, max(d for _, d in g.degree()))

    H = 128   # GraphSAGE hidden dimension
    K = 8     # number of “virtual prefix tokens”

    for model_code in COMPATIBLE_MODELS.keys():

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

        encoder = GraphSAGEEncoder(
            max_degree=max_deg,
            node_embed_dim=32,
            hidden_dim=H,
            num_layers=2,
            num_clusters=K,
            output_dim=llm_dim
        ).to(device)


        encoder.load_state_dict(torch.load(
            MODELS_PATH + 'encoder_' + model_code + '.pth',
            map_location=device
        ))

        encoder.eval()

        size_counts = defaultdict(int)

        metrics_by_size_with_prefix = defaultdict(lambda: {"acc": 0, "feas": 0, "hall": 0})
        metrics_by_size_base = defaultdict(lambda: {"acc": 0, "feas": 0, "hall": 0})
        for batch_data, prompts, solutions in test_loader:

            batch_data = batch_data.to(device)

            with torch.no_grad():
                # Compute prefix
                prefix_embeds = encoder(batch_data)           # [B, K, 128]
                            
                prompts_with_cue = [p + "\nSolution:" for p in prompts]  # length = B = 1

                # Tokenize prompt
                toks = tokenizer(prompts_with_cue, return_tensors="pt", padding=False)
                input_ids = toks.input_ids.to(device)
                prompt_embeds = base_model.get_input_embeddings()(input_ids)  # [B, M, 768]
                prompt_mask = toks.attention_mask.to(device)

                # Prepend soft prefix
                full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)    # [B, K+M, 768]
                full_mask   = torch.cat([torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device),
                                        prompt_mask], dim=1)

                # Call generate on base model directly
                base_out = base_model.generate(
                    input_ids = input_ids,
                    attention_mask = prompt_mask,
                    do_sample = False,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    stopping_criteria = stop_crit,
                )

                # Generate with prefix
                prefix_out = base_model.generate(
                    inputs_embeds = full_embeds,
                    attention_mask = full_mask,
                    do_sample = False,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    stopping_criteria = stop_crit,
                    max_length = 2048
                )

            for i in range(len(solutions)):
                num_nodes = len(solutions[i]) - 1
                size_counts[num_nodes] += 1
                # Decode generated ids to final string
                
            print()
            if prefix_out.shape[0] != base_out.shape[0]:
                raise ValueError("Batch size mismatch for base and prefix models.")
            for raw_prefix_embeds, raw_base_embeds in zip(prefix_out, base_out):
                for raw_embeds, metrics in zip([raw_prefix_embeds, raw_base_embeds], [metrics_by_size_with_prefix, metrics_by_size_base]):
                    raw_text = tokenizer.decode(raw_embeds, skip_special_tokens=True)
                    print()
                    if "]" in raw_text:
                        truncated = raw_text[: raw_text.find("]") + 1]
                        truncated = truncated.strip()
                        truncated = truncated.replace("[", "").replace("]", "")
                        truncated = truncated.split(", ")

                        duplicate_nodes = sum([truncated.count(ele) - 1 for ele in set(truncated)])

                        if sum(1 for token in truncated if token not in solutions[i]) > 0:
                            metrics[num_nodes]["hall"] += 1
                            print(f"Hallucinating: made up a node.")
                        elif len(truncated) != len(solutions[i]):
                            metrics[num_nodes]["hall"] += 1
                            print(f"Hallucinating: wrong lengths.")
                        elif duplicate_nodes > 1:
                            metrics[num_nodes]["hall"] += 1
                            print(f"Hallucinating: duplicate nodes.")
                        else:
                            
                            if truncated == solutions[i] or truncated[::-1] == solutions[i]:
                                metrics[num_nodes]["acc"] += 1
                                print(f"Correct: solution found.")
                            elif truncated[-1] == truncated[0]:
                                metrics[num_nodes]["feas"] += 1
                                print(f"Feasible: last node == first node.")
                            else:
                                metrics[num_nodes]["hall"] += 1
                                print(f"Hallucinating: for some other reason.")

                        print('GT:  ', solutions[i])
                        print('PRED:', truncated)
                        
                    else:
                        metrics[num_nodes]["hall"] += 1
                        print(f"Hallucinating: wrong format.")
                        print('GT:  ', solutions[i])
                        print('PRED:', raw_text)
                    print()
                        
        for metrics in [metrics_by_size_with_prefix, metrics_by_size_base]:
            print('-' * 100)
            print(f"Model: {model_name} {'base' if metrics == metrics_by_size_base else 'with prefix'}")
            for size in sorted(size_counts.keys()):
                print(f"Size {size}: {size_counts[size]}")
                print(f"Accuracy: {metrics[size]['acc'] / size_counts[size]:.4f}")
                print(f"Feasibility: {metrics[size]['feas'] / size_counts[size]:.4f}")
                print(f"Hallucination: {metrics[size]['hall'] / size_counts[size]:.4f}")
            print('-' * 100)
            print()

if __name__ == "__main__":
    inference()
