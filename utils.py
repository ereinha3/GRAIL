from transformers import StoppingCriteria, AutoModelForCausalLM, AutoTokenizer
import torch
from const import COMPATIBLE_MODELS, K, B

class StopOnBracket(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores=None, **kwargs):
        # if last token decodes to ']', stop
        return self.tokenizer.decode([input_ids[0, -1].item()]).strip() == "]"
    
def get_global_max_prompt_length(dataset):
    longest_input = 0
    for model_code in COMPATIBLE_MODELS.keys():
        tokenizer = AutoTokenizer.from_pretrained(COMPATIBLE_MODELS[model_code])
        model = AutoModelForCausalLM.from_pretrained(COMPATIBLE_MODELS[model_code])

        max_length = get_max_prompt_length(dataset, tokenizer, K)
        max_tokens = tokenizer.model_max_length
        max_input = model.config.max_position_embeddings

        print(f"Max prompt length for {model_code}: {max_length}")
        print(f"Max input tokens for {model_code} tokenizer:", max_tokens)
        print(f"Positional embedding size for {model_code} input:", max_input)

        if max_length > max_tokens or max_length > max_input:
            raise ValueError(f"Max input tokens for {model_code} is too small.")
        longest_input = max(longest_input, max_length)

        del model, tokenizer
        torch.cuda.empty_cache()

    return longest_input

def get_max_prompt_length(dataset, tokenizer, prefix_count):
    max_length = 0
    for problem in dataset:
        problem_with_cue = add_solution_to_prompt(problem['problem_text'], problem['path'])
        embeddings = tokenizer.encode(problem_with_cue)
        max_length = max(max_length, len(embeddings) + prefix_count)
            
    return max_length

def add_solution_to_prompt(prompt, solution):
    solution = [str(s) for s in solution]
    sol_str = "[" + ", ".join(solution) + "]"
    full_text = prompt + '\nSolution: ' + sol_str
    return full_text

def add_solution_cue_to_prompt(prompt):
    return prompt + '\nSolution: '

def is_compatible(model_name):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Tokenize a dummy prompt
    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    # Get embedding layer and create fake soft prefix
    with torch.no_grad():
        embeds = model.get_input_embeddings()(input_ids)  # [1, T, D]
        dummy_prefix = torch.zeros((1, 5, embeds.shape[-1]))  # [1, 5, D]
        inputs_embeds = torch.cat([dummy_prefix, embeds], dim=1)
        full_attention_mask = torch.cat([
            torch.ones((1, 5), dtype=torch.long),  # prefix mask
            attention_mask
        ], dim=1)

        # Try to generate from embedded input
        try:
            out = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=10,
                do_sample=False
            )
            print(prompt, tokenizer.decode(out[0]))
            return True

        except Exception as e:
            print(f"Compatibility error for {model_name}: {e}")
            return False

def print_model_input(full_texts, enc_prompts, enc_full, labels, prompt_len, tokenizer):
    for i in range(B):
        print('='*100)
        print('Full prompt')
        print('\t', full_texts[i])
        print()
        print('Encoded prompts:')
        print('\t', enc_prompts.input_ids[i, :30])
        print('\t', tokenizer.decode(enc_prompts.input_ids[i, :30]))
        print('\t', enc_prompts.input_ids[i, -30:])
        print('\t', tokenizer.decode(enc_prompts.input_ids[i, -30:]))
        print()
        print('Encoded full:')
        print('\t', enc_full.input_ids[i, :30])
        print('\t', tokenizer.decode(enc_full.input_ids[i, :30]))
        print('\t', enc_full.input_ids[i, -30:])
        print('\t', tokenizer.decode(enc_full.input_ids[i, -30:]))
        print()
        print('Labels:')
        print('\t', labels[i, :30])
        print('\t These should all be -100')
        print('\t', labels[i, K + prompt_len:])
        print('\t', tokenizer.decode(labels[i,  K + prompt_len:]))
        print('='*100)
