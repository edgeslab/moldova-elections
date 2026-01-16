# Political Narratives in the 2024 Moldovan Presidential Election: Stance, Coordination, and Influence

This repository contains data collection, preprocessing, and analysis code for studying online discourse around the Moldova elections, with a focus on stance expression, coordination, and causal contagion effects.

## Repository Structure

### `causal_analysis/`
Contains all code and experiments related to causal inference, in particular the application of Proximal Embeddings (ProEmb) to estimate stance contagion effects in the presence of latent homophily.

- **`ProEmb/`**: Implementation and configuration of the ProEmb framework.
- **`analysis.ipynb`**: Notebook for running and inspecting causal analyses, including ATE and ITE estimation.

### `coordination_analysis/`
Code for analyzing coordinated behavior among users.

- **`coordination.ipynb`**: Exploratory and quantitative analysis of coordination patterns.

### `dataset/`
Data files used across analyses.

- **`moldova_anonymized.csv`**: Main anonymized dataset containing postsId and platform..

### `stance_analysis/`
Scripts and notebooks related to political stance detection and stance-based analysis.

- **`run_stance_llm.py`**: Script for running stance classification using large language models.
- **`stances.ipynb`**: Analysis and validation of stance labels and distributions.

### `topic_analysis/`
Topic modeling and thematic analysis of election-related discourse.

- **`topics.ipynb`**: Topic extraction, inspection, and visualization.

### Top-level files

- **`analysis.ipynb`**: High-level exploratory analysis.
- **`collection_scripts.txt`**: Query scripts used for data collection from the external data provider.
- **`README.md`**: This file.

## Notes

- Analyses are organized by methodological focus (causal, coordination, stance, topic).
- Notebooks are primarily intended for exploration and result reproduction.

