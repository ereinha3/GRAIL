from encoders import SAGEEncoder, GATEncoder, GINEncoder

MODELS_PATH = 'saved_models/'
OUTPUT_PATH = 'outputs/'

COMPATIBLE_MODELS = {
    'opt125': "facebook/opt-125m",
    'pythia160': "EleutherAI/pythia-160m",
    'gptneo125': "EleutherAI/gpt-neo-125M",
    'gpt2': "gpt2"
}

#     'gptneo125': "EleutherAI/gpt-neo-125M",


ENCODERS = {
    'SAGEEncoder': SAGEEncoder,
    'GATEncoder': GATEncoder,
    'GINEncoder': GINEncoder,
    'None': None
}


H = 128   # hidden dim for encoder
K = 8   # number of clusters (METIS blocks / prefix tokens)
B = 8     # batch size


MAX_INPUT_LENGTH = 1050
# Empircally derived from the longest input function in utils.py and padded up just a bit