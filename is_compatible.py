from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose one of the confirmed-compatible models
model_name = "EleutherAI/pythia-160m" 

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
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        max_new_tokens=10,
        do_sample=False
    )

print(prompt, tokenizer.decode(out[0]))
