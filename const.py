MODELS_PATH = 'saved_models/'

COMPATIBLE_MODELS = {
    'opt125': "facebook/opt-125m",
    'pythia160': "EleutherAI/pythia-160m",
    'gptneo125': "EleutherAI/gpt-neo-125M"
}


H = 128   # hidden dim for GraphSAGE
K = 8     # number of clusters (METIS blocks / prefix tokens)
B = 8     # batch size


MAX_INPUT_LENGTH = 1800