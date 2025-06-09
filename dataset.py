from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
import torch

class TSPDataset(Dataset):
    """
    A Dataset wrapping GraphArena TSP problems, returning for each index:
      - A PyG Data object (with edge_index, optional edge_attr, deg_idx, batch)
      - The prompt text (string)
      - The solution path (list of ints)
    """
    def __init__(self, problems):
        self.problems = problems
        # Pre-convert all NetworkX graphs to PyG Data once
        self.data_list = []
        for prob in problems:
            nx_g = prob['graph']
            # Pull in edge weights if available
            data = from_networkx(nx_g, group_edge_attrs=['weight'])
            # Degree index feature
            deg_idx = torch.tensor([d for _, d in nx_g.degree()], dtype=torch.long)
            data.deg_idx = deg_idx
            # Dummy batch vector (will be overwritten by Batch.from_data_list)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            self.data_list.append(data)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        # Return (Data, prompt_text, solution_path)
        data = self.data_list[idx]
        prompt = self.problems[idx]['problem_text']
        id_path   = self.problems[idx]['path']                         # e.g. [944, 2222, 654, 1496, 944]
        nx_g      = self.problems[idx]['graph']
        code_path = [nx_g.nodes[n]['name'] for n in id_path]
        return data, prompt, code_path

def collate_fn(batch):
    """
    Custom collate to batch PyG Data objects and gather prompts + solutions lists.
    batch is a list of tuples (Data, prompt, solution)
    """
    data_list, prompts, solutions = zip(*batch)
    # Create a batched Data object
    batch_data = Batch.from_data_list(data_list)
    return batch_data, list(prompts), list(solutions)