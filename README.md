# URL_Citation_Classification_Intermediate
The source code for the paper "On an Intermediate Task for Classifying URL Citations on Scholarly Papers" in LREC-COLING 2024

# Setup
This repository assumes `uv run` for command execution.

Example:

```bash
uv sync
```

# Source Code
All programs used in experiments are in ```/src``` directory.
- training.py: Main component for training the model (used for both simple fine-tuning and our method)
- *_run.py: The program to load and preprocess the dataset of each task (e.g., cola_run.py -> The program for CoLA)
  - url_cite_run.py: Main script for our method and Tsunokake and Matsubara (2022)'s method
  - url_zhao_run.py: The script for Zhao et al. (2019)'s method
  - others: Scripts for Section "4.3.5. Effectiveness of Our Method for Other Text Classification Tasks"

# Hydra + Weights & Biases
`src/base_hydra` now supports W&B experiment tracking through Hydra config.

Example:

```bash
uv run python src/base_hydra/src/url_cite_run_hydra.py wandb.enabled=true wandb.project=url-citation-classification
```

Useful overrides:

```bash
uv run python src/base_hydra/src/url_cite_run_hydra.py \
  model=modernbert_base \
  training=debug \
  wandb.enabled=true \
  wandb.project=url-citation-classification \
  wandb.tags='[debug,modernbert]'
```

If you want offline logging, set `wandb.mode=offline`.
