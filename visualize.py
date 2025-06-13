import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def process_data(data):
    # Initialize lists to store processed data
    processed_data = []
    
    for model in data:
        for encoder in data[model]:
            for size in data[model][encoder]:
                entry = data[model][encoder][size]
                
                # Calculate total hallucination
                hall_total = sum(entry['hall'].values())
                
                processed_data.append({
                    'model': model,
                    'encoder': encoder,
                    'size': int(size),
                    'accuracy': entry['acc'],
                    'feasibility': entry['feas'],
                    'hallucination': hall_total,
                    'wrong_format': entry['hall']['wrong_format'],
                    'wrong_length': entry['hall']['wrong_length'],
                    'duplicate_nodes': entry['hall']['duplicate_nodes'],
                    'made_up_node': entry['hall']['made_up_node']
                })
    
    return pd.DataFrame(processed_data)

def create_encoder_plots(df, encoder, output_dir, fig_num):
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_theme()
    
    # Create a 2x2 figure
    fig = plt.figure(figsize=(20, 16))
    
    # Group by model and encoder, calculate mean metrics
    metrics = df.groupby(['model', 'encoder'])[['accuracy', 'feasibility', 'hallucination']].mean()
    metrics = metrics.reset_index()
    
    # Melt the dataframe for easier plotting
    metrics_melted = pd.melt(metrics, 
                            id_vars=['model', 'encoder'],
                            value_vars=['accuracy', 'feasibility', 'hallucination'],
                            var_name='metric',
                            value_name='value')
    
    # Plot 1: Metrics
    ax1 = plt.subplot(2, 2, 1)
    sns.barplot(data=metrics_melted[metrics_melted['encoder'] == encoder],
                x='model', y='value', hue='metric', ax=ax1)
    ax1.set_title(f'{encoder} Metrics')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Model')
    
    # Plot 2: Hallucination Types
    ax2 = plt.subplot(2, 2, 2)
    hall_types = df.groupby(['model', 'encoder'])[['wrong_format', 'wrong_length', 'duplicate_nodes', 'made_up_node']].mean()
    hall_types = hall_types.reset_index()
    hall_melted = pd.melt(hall_types[hall_types['encoder'] == encoder],
                         id_vars=['model'],
                         value_vars=['wrong_format', 'wrong_length', 'duplicate_nodes', 'made_up_node'],
                         var_name='hallucination_type',
                         value_name='count')
    sns.barplot(data=hall_melted, x='model', y='count', hue='hallucination_type', ax=ax2)
    ax2.set_title(f'{encoder} Hallucination Types')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Model')
    
    # Plot 3: Metrics vs Problem Size
    ax3 = plt.subplot(2, 2, 3)
    metrics_size = df[df['encoder'] == encoder].groupby(['model', 'size'])[['accuracy', 'feasibility', 'hallucination']].mean()
    metrics_size = metrics_size.reset_index()
    metrics_size_melted = pd.melt(metrics_size,
                                 id_vars=['model', 'size'],
                                 value_vars=['accuracy', 'feasibility', 'hallucination'],
                                 var_name='metric',
                                 value_name='value')
    sns.lineplot(data=metrics_size_melted, x='size', y='value', hue='metric', style='model', ax=ax3)
    ax3.set_title(f'{encoder} Metrics vs Problem Size')
    ax3.set_ylabel('Value')
    ax3.set_xlabel('Problem Size')
    
    # Plot 4: Hallucination Types vs Problem Size
    ax4 = plt.subplot(2, 2, 4)
    hall_size = df[df['encoder'] == encoder].groupby(['model', 'size'])[['wrong_format', 'wrong_length', 'duplicate_nodes', 'made_up_node']].mean()
    hall_size = hall_size.reset_index()
    hall_size_melted = pd.melt(hall_size,
                              id_vars=['model', 'size'],
                              value_vars=['wrong_format', 'wrong_length', 'duplicate_nodes', 'made_up_node'],
                              var_name='hallucination_type',
                              value_name='count')
    sns.lineplot(data=hall_size_melted, x='size', y='count', hue='hallucination_type', style='model', ax=ax4)
    ax4.set_title(f'{encoder} Hallucination Types vs Problem Size')
    ax4.set_ylabel('Count')
    ax4.set_xlabel('Problem Size')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'encoder_metrics_{fig_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(df, output_dir):
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get unique encoders
    encoders = df['encoder'].unique()
    
    # Create plots for each encoder
    for i, encoder in enumerate(encoders):
        create_encoder_plots(df, encoder, output_dir, i+1)
    
    # Create a comparison plot across all encoders
    plt.figure(figsize=(20, 16))
    
    # Group by model and encoder, calculate mean metrics
    metrics = df.groupby(['model', 'encoder'])[['accuracy', 'feasibility', 'hallucination']].mean()
    metrics = metrics.reset_index()
    
    # Melt the dataframe for easier plotting
    metrics_melted = pd.melt(metrics, 
                            id_vars=['model', 'encoder'],
                            value_vars=['accuracy', 'feasibility', 'hallucination'],
                            var_name='metric',
                            value_name='value')
    
    # Create comparison plot
    g = sns.catplot(data=metrics_melted,
                    x='model',
                    y='value',
                    hue='metric',
                    col='encoder',
                    kind='bar',
                    height=5,
                    aspect=1.5)
    
    g.fig.suptitle('Comparison of Metrics Across All Encoders', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_encoders_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def make_results_table(df, output_dir):
    # Aggregate mean for each (model, encoder) pair
    grouped = df.groupby(['model', 'encoder']).mean(numeric_only=True)
    # Prepare multi-index rows
    index = [
        ('accuracy', ''),
        ('feasibility', ''),
        ('hallucination', ''),
        ('hallucination', 'wrong_format'),
        ('hallucination', 'wrong_length'),
        ('hallucination', 'duplicate_nodes'),
        ('hallucination', 'made_up_node'),
    ]
    columns = []
    for model, encoder in grouped.index:
        columns.append((model, encoder))
    table = pd.DataFrame(index=pd.MultiIndex.from_tuples(index, names=['metric', 'subtype']), columns=pd.MultiIndex.from_tuples(columns, names=['model', 'encoder']))
    for (model, encoder), row in grouped.iterrows():
        table.loc[('accuracy', ''), (model, encoder)] = row['accuracy']
        table.loc[('feasibility', ''), (model, encoder)] = row['feasibility']
        table.loc[('hallucination', ''), (model, encoder)] = row['hallucination']
        table.loc[('hallucination', 'wrong_format'), (model, encoder)] = row['wrong_format']
        table.loc[('hallucination', 'wrong_length'), (model, encoder)] = row['wrong_length']
        table.loc[('hallucination', 'duplicate_nodes'), (model, encoder)] = row['duplicate_nodes']
        table.loc[('hallucination', 'made_up_node'), (model, encoder)] = row['made_up_node']
    # Save as CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    table.to_csv(output_dir / 'results_table.csv')
    # Print pretty version
    print('\n===== Results Table (mean over all node counts) =====')
    print(table.round(3).fillna(''))

def plot_metric_vs_nodecount(df, metric, output_dir, pad_top=False):
    plt.figure(figsize=(10, 7))
    df['pair'] = df['model'] + ' | ' + df['encoder']
    ymax = 0
    for pair, subdf in df.groupby('pair'):
        subdf = subdf.sort_values('size')
        y = subdf[metric]
        plt.plot(subdf['size'], y, marker='o', label=pair)
        if pad_top:
            ymax = max(ymax, y.max())
    plt.xlabel('Node Count')
    plt.ylabel(f'{metric.replace("_", " ").capitalize()}')
    plt.title(f'{metric.replace("_", " ").capitalize()} vs Node Count')
    if pad_top:
        plt.ylim(0, min(1.05, ymax * 1.05 + 0.01))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{metric}_vs_nodecount.png', dpi=300)
    plt.close()

def plot_all_metrics_vs_nodecount(df, output_dir):
    plot_metric_vs_nodecount(df, 'accuracy', output_dir)
    plot_metric_vs_nodecount(df, 'feasibility', output_dir)
    for hall_type in ['wrong_format', 'wrong_length', 'duplicate_nodes', 'made_up_node']:
        plot_metric_vs_nodecount(df, hall_type, output_dir, pad_top=True)

def main():
    # Load and process data
    data = load_data('outputs/accuracy.json')
    df = process_data(data)
    
    # Create visualizations
    plot_metrics(df, 'outputs/plots')
    make_results_table(df, 'outputs/plots')
    plot_all_metrics_vs_nodecount(df, 'outputs/plots')
    
    print("Visualizations and table have been saved to outputs/plots/")

if __name__ == "__main__":
    main()
