# X-Node: Self Explanation is All We Need

## Overview

XNode is a framework for explainable graph neural networks that combines topological feature analysis with language model-based explanations. The system uses Grok API to generate human-readable explanations of graph topological features and their relationship to node classification. 

#### Accepted in GRAIL, MICCAI 2025 Conference (ORAL) 🎉

Youtube Presentation: [https://www.youtube.com/watch?v=pdUGQr6dnt4&t=361s](https://www.youtube.com/watch?v=pdUGQr6dnt4&t=361s)
Arxiv Link: [https://arxiv.org/abs/2508.10461](https://arxiv.org/abs/2508.10461)


## Architecture
<img width="621" height="380" alt="image" src="https://github.com/user-attachments/assets/32af3c8e-abdf-4943-9747-484408a4b439" />

## Project Structure

```
XNode/
├── data_processing/     # Graph construction and data preprocessing
├── models/             # Baseline GNN models (GCN, GAT, GIN)
├── reasoners/          # Reasoner models with language explanations
├── utils/              # Utility functions and helpers
├── datasets/           # Pre-built graph datasets
├── results/            # Output results and visualizations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Key Features

- **Graph Construction**: Converts MedMNIST image datasets to k-NN graphs using ResNet18 features
- **Baseline Models**: Standard GNN implementations (GCN, GAT, GIN)
- **Reasoner Models**: Enhanced GNNs with explanation generation capabilities
- **Topological Analysis**: Computes graph-theoretic features for context
- **Language Explanations**: Uses Grok API to generate human-readable explanations
- **Multiple Datasets**: Support for OrganCMNIST, BloodMNIST, TissueMNIST, and MorphoMNIST

## Installation

```bash
pip install -r requirements.txt
```

## Datasets
### MedMNIST
A lightweight benchmark for biomedical image classification, covering organ, tissue, and blood datasets.

- **Homepage:** [https://medmnist.com/](https://medmnist.com/)
- **Install via pip:**
```bash
  pip install medmnist
```

## Usage

### 1. Data Processing

Build graphs from MedMNIST datasets:

```bash
python data_processing/build_graphs.py --dataset organcmnist
python data_processing/build_graphs.py --dataset bloodmnist
python data_processing/build_graphs.py --dataset tissuemnist
```

### 2. Baseline Models

Train baseline GNN models:

```bash
python models/train_baseline.py --model gcn --dataset organcmnist
python models/train_baseline.py --model gat --dataset organcmnist
python models/train_baseline.py --model gin --dataset organcmnist
```

### 3. Reasoner Models

Train reasoner models with explanations:

```bash
python reasoners/train_reasoner.py --model gcn --dataset organcmnist
python reasoners/train_reasoner.py --model gat --dataset organcmnist
python reasoners/train_reasoner.py --model gin --dataset organcmnist
```

### 4. Generate Explanations

Generate and save explanations:

```bash
python reasoners/generate_explanations.py --model gcn --dataset organcmnist
```

### Quick Demo (recommended)

Run a fast end-to-end example with the bundled Organcmnist graph:

```bash
python quick_start.py
```

### Full Pipeline (train + explain)

Fast smoke test for GCN on Organcmnist (uses offline explanations, no API key needed):

```bash
python run_pipeline.py \
  --dataset organcmnist \
  --models gcn \
  --train_baselines \
  --train_reasoners \
  --generate_explanations \
  --offline \
  --fast \
  --epochs 5 \
  --n_folds 2
```

To use API-backed explanations instead of offline mode:

```bash
export GROQ_API_KEY="YOUR_API_KEY_HERE"
python run_pipeline.py \
  --dataset organcmnist \
  --models gcn \
  --train_baselines \
  --train_reasoners \
  --generate_explanations \
  --api_key "$GROQ_API_KEY" \
  --fast \
  --epochs 5 \
  --n_folds 2
```

## Topological Features

The system computes the following topological features for each node:

- **Degree**: Number of connections
- **Clustering Coefficient**: Local clustering measure
- **Two-hop Agreement**: Label consistency in 2-hop neighborhood
- **Eigenvector Centrality**: Global importance measure
- **Degree Centrality**: Local importance measure
- **Average Edge Weight**: Mean similarity to neighbors

## Explanation Types

1. **Topological Context**: Analysis of node's position in graph structure
2. **Feature Importance**: Which topological features influence classification
3. **Neighborhood Analysis**: How local structure affects predictions
4. **Global Patterns**: Graph-wide structural insights

## Datasets

- **OrganCMNIST**: Medical organ classification
- **BloodMNIST**: Blood cell classification  
- **TissueMNIST**: Tissue type classification
- **MorphoMNIST**: Morphological digit variants

## Results

Results are saved in the `results/` directory with:
- Training curves and metrics
- t-SNE visualizations
- Explanation outputs
- Model checkpoints
<img width="582" height="232" alt="image" src="https://github.com/user-attachments/assets/5bd391d7-7e26-4445-af35-96885dc4bac8" />

<img width="618" height="475" alt="image" src="https://github.com/user-attachments/assets/bc032157-386c-4ea3-98d7-1fa818b6e848" />


## Configuration

Set your Grok API key in environment variables:
```bash
export GROQ_API_KEY="your_api_key_here"
```

## Citation

If you use this code in your research, please cite:
```
@misc{sengupta2025xnodeselfexplanationneed,
      title={X-Node: Self-Explanation is All We Need}, 
      author={Prajit Sengupta and Islem Rekik},
      year={2025},
      eprint={2508.10461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.10461}, 
}
``` 
