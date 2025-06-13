#!/usr/bin/env bash

models=('pythia160' 'opt125')
encoders=('SAGEEncoder' 'GATEncoder' 'GINEncoder')

# Iterate over each combination
for model in "${models[@]}"; do
  for encoder in "${encoders[@]}"; do
    python inference.py --model_code "$model" --encoder "$encoder"
  done
done
