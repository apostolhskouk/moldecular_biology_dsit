# Navigating Chemical Space with Latent Flows

This repository contains the code for reproducing and extending the paper [Navigating Chemical Space with Latent Flows](https://openreview.net/forum?id=aAaV4ZbQ9j), along with additional extensions and optimization methods.

## Dependencies

This project builds upon three repositories:

- **[ChemFlow](https://github.com/garywei944/ChemFlow)** - Original implementation code
- **[MolTransformer](https://github.com/baskargroup/MolTransformer_repo)** - Transformer-based optimization methods for extensions
- **[AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina)** - Molecular docking score evaluation

## Setup

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate dsit_chemflow
```

### Data and Checkpoints (Optional)

To download the processed data, pre-trained models, intermediate results, and final results:

```bash
huggingface-cli download ApostolosK/chemflow-assets --include "ChemFlow/*" --local-dir .
```

With the downloaded assets, you can immediately run the results display scripts. Otherwise, you can train the models from scratch (note: this will take considerable time).

## Usage

All instructions for running experiments and displaying results are provided in `deliverable.ipynb`. 

1. Open the notebook using the created conda environment
2. Follow the step-by-step instructions
3. Explore the demo at the end showcasing molecular structure optimization with different methods

The notebook includes comprehensive guidance for both reproducing the original paper results, running the extended optimization approaches and displaying the results.
