#!/usr/bin/env python3
"""
Quick start script for XNode framework.
Runs a simple example with OrganCMNIST dataset.
"""

import os
import sys
import subprocess

def main():
    print("XNode Quick Start")
    print("=" * 50)
    
    # Check if we have the required graph file
    graph_file = "datasets/G_Organcmnist_inductive.gpickle"
    
    if not os.path.exists(graph_file):
        print(f"Graph file {graph_file} not found!")
        print("Please ensure you have the required graph files in the datasets/ directory.")
        print("You can copy them from the S-XAI/Models/ directory or build them using:")
        print("python data_processing/build_graphs.py --dataset organcmnist")
        return
    
    print(f"Found graph file: {graph_file}")
    
    # Run a simple baseline training
    print("\nTraining GCN baseline model...")
    cmd = f"python models/train_baseline.py --model gcn --dataset organcmnist --graph_file {graph_file} --output_dir results --epochs 50 --n_folds 2"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ GCN baseline training completed!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"✗ Error training GCN baseline: {e}")
        if e.stderr:
            print("Error:", e.stderr)
        return
    
    # Run reasoner training
    print("\nTraining GCN reasoner model...")
    cmd = f"python reasoners/train_reasoner.py --model gcn --dataset organcmnist --graph_file {graph_file} --output_dir results --epochs 50 --n_folds 2"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ GCN reasoner training completed!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"✗ Error training GCN reasoner: {e}")
        if e.stderr:
            print("Error:", e.stderr)
        return
    
    print("\n" + "=" * 50)
    print("Quick start completed!")
    print("Check the results/ directory for outputs.")
    print("\nTo run the full pipeline, use:")
    print("python run_pipeline.py --dataset organcmnist --train_baselines --train_reasoners")
    print("\nTo generate explanations (requires API key):")
    print("python run_pipeline.py --dataset organcmnist --generate_explanations --api_key YOUR_API_KEY")

if __name__ == '__main__':
    main() 