# ProEmb: Contagion Effect Estimation Using Proximal Embeddings

This package implements the ProEmb framework for estimating contagion effects using proximal embeddings, as described in the paper by Fatemi and Zheleva (2023).

## Overview

ProEmb learns a latent representation of high-dimensional proxy features to estimate the causal effect of social contagion. It uses:

* A **Variational Autoencoder (VAE)** to embed proxy features.
* An **adversarial discriminator** to balance treatment and control distributions.
* A **T-learner meta-learner** to predict potential outcomes and compute Average Contagion Effect (ACE).

The full pipeline includes two phases:

1. **Training Phase:** Learns balanced latent embeddings and outcome prediction models.
2. **Inference Phase:** Uses trained models to compute counterfactual outcomes, ICE, and ACE.

## Folder Structure

```
proemb/
├── __init__.py
├── models/                       # Core model components
│   ├── __init__.py
│   ├── vae_encoder.py            # Encoder network
│   ├── vae_decoder.py            # Decoder network
│   ├── sampler.py                # VAE sampler (reparameterization trick)
│   ├── discriminator.py          # Adversarial discriminator
│   └── meta_learner.py           # T-learner with pluggable base learners
│
├── loss/                       # Loss functions
│   ├── __init__.py
│   ├── losses.py                 # VAE loss (reconstruction + KL) and Balance regularizer 
│
├── engines/                       # Training and inference logic
│   ├── __init__.py
│   ├── trainer.py                # Training pipeline
│   └── inference.py              # Inference pipeline
│
├── data/                         # Example data (or data loaders/utilities)
│   ├── train_dataset.csv         # Training CSV with proxies, treatment, outcome
│
├── pipeline.py                   # Example pipeline for training and inference
│
└── README.md
```

## CSV Format

Input CSVs must contain:

* `ego_feat0`, `ego_feat1`, ..., `ego_featN`: Features of ego
* `ngb_feat0`, `ngb_feat1`, ..., `ngb_featN`: Features of peer(s)
* `treatment`: Indicates neighbor outcome at t-1 (0 or 1)
* `outcome`: Ego outcome at time t (only for training)

Test files must contain only the proxy features.

## How to Run the Pipeline

### 1. Install Dependencies

```bash
pip install torch pandas lightgbm
```

### 2. Prepare Data

Ensure your CSV files are placed in the `proemb/data/` folder and follow the column naming scheme described above.

### 3. Run Training and Inference

```bash
python proemb/pipeline.py
```

This will:

* Train the VAE, discriminator, and meta-learner
* Print the Average Contagion Effect (ACE)
* Print a few sample Individual Contagion Effects (ITE)

## Customization

* Change latent dimension, learning rates, or network sizes in `Trainer()` inside `pipeline.py`.
* Use different base learners (`mlp`, `linear`, or `lightgbm`) via `base_model` argument.
* Extend to other tasks like synthetic experiments or multi-hop graphs as needed.

## Reference

Fatemi, Z., & Zheleva, E. (2025, June). Contagion Effect Estimation Using Proximal Embeddings. In Causal Learning and Reasoning (pp. 243-259). PMLR.

### BibTeX 

@inproceedings{fatemi2025contagion, \
  title={Contagion Effect Estimation Using Proximal Embeddings}, \
  author={Fatemi, Zahra and Zheleva, Elena}, \
  booktitle={Causal Learning and Reasoning}, \
  pages={243--259}, \
  year={2025}, \
  organization={PMLR} \
}
