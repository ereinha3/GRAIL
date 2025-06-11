import networkx as nx
import random
import pickle
import math
import numpy as np
import fast_tsp

class TSP_Task:
    """
    Generates:
      - `num_train` random metric TSPs of size between [min_nodes..max_nodes],
      - `num_test` random metric TSPs of the same size range.
    Each problem is stored as:
      { 'id', 'problem_text', 'graph', 'exact_answer', 'path' }.
    Train set is saved to: dataset/TSP_random_<min>_<max>_train_<num_train>.pkl
    Test set is saved to:  dataset/TSP_random_<min>_<max>_test_<num_test>.pkl
    """

    def __init__(self,
                 num_train=2048,
                 num_test=256,
                 num_val=256,
                 data_loc='dataset'):
        self.easy_min_nodes  = 4
        self.easy_max_nodes  = 9
        self.hard_min_nodes  = 10
        self.hard_max_nodes  = 15
        self.num_train       = num_train
        self.num_test        = num_test
        self.num_val         = num_val
        self.data_loc        = data_loc
        self.problem_set = {
            'easy': {
                'train': [],
                'test': [],
                'val': []
            },
            'hard': {
                'train': [],
                'test': [],
                'val': []
            }
        }
        self.data_path = f"{self.data_loc}/TSP.pkl"

    def compute_tour_length(self, graph, route):
        """Sum up graph[u][v]['weight'] over consecutive edges in route."""
        total = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            total += graph[u][v]['weight']
        return total

    def exact_solver(self, graph):
        """
        Build n×n integer distance matrix from graph, solve TSP exactly,
        return (length, closed_loop_route).
        """
        n = graph.number_of_nodes()
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                w = graph[i][j]['weight']
                dist[i][j] = w
                dist[j][i] = w

        route = fast_tsp.solve_tsp_exact(dist)  # returns a list of length n
        route.append(route[0])  # close the loop
        length = self.compute_tour_length(graph, route)
        return length, route

    def _build_one_instance(self, pid, difficulty):
        """
        Build and solve one random TSP instance with node_count ∈ [min_nodes..max_nodes].
        Returns exactly the dictionary to append to train/test list.
        """
        if difficulty == 'easy':
            n = random.randint(self.easy_min_nodes, self.easy_max_nodes)
        elif difficulty == 'hard':
            n = random.randint(self.hard_min_nodes, self.hard_max_nodes)
        else:
            raise ValueError(f"Invalid difficulty: {difficulty}")

        # 2) Generate n random points in the 100x100 square
        points = [(random.randint(1, 100), random.randint(1, 100)) for _ in range(n)]

        # 3) Build a complete metric graph H on n nodes
        H = nx.Graph()
        for i in range(n):
            H.add_node(i, name=f"A{i}")
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                d = int(round(math.hypot(xi - xj, yi - yj)))
                H.add_edge(i, j, weight=d)

        # 4) Solve exactly for the shortest tour
        exact_length, exact_path = self.exact_solver(H)

        # 5) Build the natural‐language prompt
        problem_text = self.generate_problem(H)

        # 6) Return the dict for this problem
        return {
            'id': pid,
            'problem_text': problem_text,
            'graph': H,
            'exact_answer': exact_length,
            'path': exact_path
        }

    def generate_dataset(self):
        """
        Build `num_train` training TSPs, then `num_test` testing TSPs,
        and write them to two separate pickle files.
        """
        # Build training set
        for pid in range(self.num_train):
            easy_inst = self._build_one_instance(pid, 'easy')
            self.problem_set['easy']['train'].append(easy_inst)
            hard_inst = self._build_one_instance(pid, 'hard')
            self.problem_set['hard']['train'].append(hard_inst)

        for pid in range(self.num_test):
            easy_inst = self._build_one_instance(pid, 'easy')
            self.problem_set['easy']['test'].append(easy_inst)
            hard_inst = self._build_one_instance(pid, 'hard')
            self.problem_set['hard']['test'].append(hard_inst)

        for pid in range(self.num_val):
            easy_inst = self._build_one_instance(pid, 'easy')
            self.problem_set['easy']['val'].append(easy_inst)
            hard_inst = self._build_one_instance(pid, 'hard')
            self.problem_set['hard']['val'].append(hard_inst)

        with open(self.data_path, "wb") as f:
            pickle.dump(self.problem_set, f)
        print(f"Saved {self.num_train} random train TSPs, {self.num_val} random val TSPs, and {self.num_test} random test TSPs to {self.data_path}")

    def load_dataset(self):
        """
        Load train & test problem sets from disk,
        assuming generate_dataset() has already been run.
        """
        try:
            with open(self.data_path, "rb") as f:
                self.problem_set = pickle.load(f)
        except FileNotFoundError:
            print(f"File {self.data_path} not found. Please run generate_dataset() first.")

        print(f"Loaded {len(self.problem_set['easy']['train'])} easy train problems from {self.data_path}")
        print(f"Loaded {len(self.problem_set['easy']['test'])} easy test problems from {self.data_path}")
        print(f"Loaded {len(self.problem_set['easy']['val'])} easy val problems from {self.data_path}")
        print(f"Loaded {len(self.problem_set['hard']['train'])} hard train problems from {self.data_path}")
        print(f"Loaded {len(self.problem_set['hard']['test'])} hard test problems from {self.data_path}")
        print(f"Loaded {len(self.problem_set['hard']['val'])} hard val problems from {self.data_path}")

    def generate_problem(self, graph):
        """
        Given a complete metric graph with node‐names "A0", "A1", …,
        format and return the natural‐language prompt listing:
          - "Airports to visit: A0, A1, …"
          - "X to Y: Z" for each edge
          - final instructions on bracket formatting.
        """
        prompt = [
            "You are required to solve the Travelling Salesman Problem for an undirected flight route network. "
            "Your objective is to determine the shortest possible route that visits each of the listed airports exactly once "
            "and returns to the starting point."
        ]
        prompt.append("\n**Problem to Solve**\n")
        prompt.append(
            "- Airports to visit: " + ", ".join([graph.nodes[i]['name'] for i in graph.nodes()])
        )
        prompt.append("- Travel distances (in kilometers) between each pair of airports:")
        for u, v, data in graph.edges(data=True):
            prompt.append(f"{graph.nodes[u]['name']} to {graph.nodes[v]['name']}: {data['weight']}")
        prompt.append("Please calculate the shortest tour and format your answer as follows: [Airport A, Airport B, ..., Airport A]")
        return "\n".join(prompt)

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    task = TSP_Task()
    task.generate_dataset()
