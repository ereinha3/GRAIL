# TSP Graph Model Training & Evaluation

Welcome to the **TSP Graph Model** repository! This project provides a full pipeline for training, evaluating, and visualizing the performance of various graph neural network (GNN) encoders and language models on the classic **Traveling Salesman Problem (TSP)**. The codebase is designed for extensibility and reproducibility, making it easy to benchmark new models or encoders on TSP-style graph tasks.

---

## Project Description

This repository explores the intersection of graph neural networks and large language models for solving combinatorial optimization problems, specifically the Traveling Salesman Problem (TSP). The pipeline includes:

- **Dataset Generation:** Automated creation of TSP instances of varying sizes and difficulties.
- **Model Training:** Training of different GNN encoders (e.g., SAGE, GAT, GIN) as graph preprocessors, paired with transformer-based language models.
- **Evaluation:** Automated inference and performance logging for all model/encoder pairs.
- **Visualization:** Generation of publication-quality plots and tables for accuracy, feasibility, and hallucination metrics, including detailed breakdowns by hallucination type.

The code is modular, GPU-accelerated, and designed for research and benchmarking.

---

## Setup

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the TSP dataset**
   ```bash
   python TSP.py
   ```

---

## Training

Train all model/encoder combinations (this may take 4–6 hours on a high-end GPU):

```bash
mkdir -p saved_models
bash train.sh
```

---

## Evaluation

Run inference and save model performance (about 30 minutes on a 4090):

```bash
mkdir -p outputs
bash eval.sh
```

---

## Results & Visualization

After inference, you can:

1. **Print results in the console**
   ```bash
   python printer.py
   ```

2. **Visualize results with graphs and tables**
   ```bash
   python visualize.py
   ```
   - Plots and tables will be saved in the `outputs/plots/` directory.
   - The script generates:
     - Accuracy and feasibility vs. node count (all model/encoder pairs)
     - Four hallucination-type plots vs. node count (all model/encoder pairs)
     - A summary table of results (`results_table.csv`)

---

## Directory Structure

```
TSP.py              # Dataset generation
train.sh            # Trains all model/encoder pairs
train.py            # Training script (called by train.sh)
eval.sh             # Runs inference for all trained models
eval.py             # Evaluation script (called by eval.sh)
printer.py          # Prints summary results
visualize.py        # Generates plots and tables
outputs/            # Stores evaluation results and plots
saved_models/       # Stores trained model checkpoints
requirements.txt    # Python dependencies
```

---

## Notes

- Training and evaluation are GPU-intensive. For best results, use a machine with a modern NVIDIA GPU.
- All scripts are designed to be run from the repository root.
- The codebase is modular—feel free to add new encoders, models, or tasks!

---

## Contact

For questions, suggestions, or contributions, please open an issue or pull request. Feel free to contact the author at ethanreinhart@gmail.com directly if needed.