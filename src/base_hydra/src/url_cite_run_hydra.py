"""
Hydra entrypoint for URL citation classification.

This module keeps the logic from the original script while exposing
configuration through Hydra.
"""

import logging
import os
import sys

# Make `src/base_hydra/src` (this directory) importable first, then `src/base`.
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "..", "base"))

# Repository-level paths.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_HYDRA_OUTPUT_BASE = os.path.join(_REPO_ROOT, "output", "urlcitation", "hydra")
os.environ.setdefault("HYDRA_OUTPUT_BASE", _HYDRA_OUTPUT_BASE)

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from preprocess_hydra import preprocess as preprocess_hydra
from training_hydra import main_hydra
from url_cite_assets import SPECIAL_TOKENS, SplitedData, TrainingConfig

log = logging.getLogger(__name__)


def load_data(
    data_path: str,
    test_seed: int,
    test_size: float,
    train_seed: int,
    model_name: str = "bert-base-uncased",
) -> SplitedData:
    """Load CSV data, split train/test, and preprocess for training."""
    resolved_data_path = data_path
    if not os.path.isabs(resolved_data_path):
        resolved_data_path = os.path.join(_REPO_ROOT, resolved_data_path)

    # Backward-compatible fallback for legacy directory naming.
    if not os.path.exists(resolved_data_path):
        fallback_data_path = os.path.join(
            _REPO_ROOT,
            "data",
            os.path.basename(data_path),
        )
        if os.path.exists(fallback_data_path):
            log.warning(
                "Data file not found at %s. Falling back to %s",
                resolved_data_path,
                fallback_data_path,
            )
            resolved_data_path = fallback_data_path
        else:
            raise FileNotFoundError(
                f"Data file not found: {resolved_data_path}. "
                f"Tried fallback path: {fallback_data_path}. "
                "Please set `data.data_path` in src/base_hydra/conf/config.yaml."
            )

    df = pd.read_csv(
        resolved_data_path,
        encoding="utf-8",
        index_col=0,
    )

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = train_test_split(
        df,
        shuffle=True,
        random_state=test_seed,
        test_size=test_size,
    )

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    preprocessed_data = preprocess_hydra(
        train_df,
        test_df,
        train_seed,
        model_name=model_name,
    )

    return {
        "train_X": preprocessed_data[0],
        "train_labels": list(preprocessed_data[1:4]),
        "valid_X": preprocessed_data[4],
        "valid_labels": list(preprocessed_data[5:8]),
        "test_X": preprocessed_data[8],
        "test_labels": list(preprocessed_data[9:12]),
    }


def cite_main(cfg: DictConfig):
    """Build TrainingConfig from Hydra config and run training."""
    output_base_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info("Hydra output dir: %s", output_base_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    data = load_data(
        data_path=cfg.data.data_path,
        test_seed=cfg.data.test_seed,
        test_size=cfg.data.test_size,
        train_seed=cfg.data.train_seed,
        model_name=cfg.model.encoder_model_name,
    )

    n_classes = [4, 10, 6]

    intermediate_training_config = OmegaConf.to_container(
        cfg.training.intermediate,
        resolve=True,
    )
    fine_tuning_config = OmegaConf.to_container(
        cfg.training.fine_tuning,
        resolve=True,
    )

    inter_patience = intermediate_training_config.pop("early_stopping_patience", 2)
    ft_patience = fine_tuning_config.pop("early_stopping_patience", 5)

    config = TrainingConfig(
        n_classes=n_classes,
        special_tokens=SPECIAL_TOKENS,
        n_sample=cfg.n_sample,
        inter_split_seed=cfg.inter_split_seed,
        training_seed=cfg.seed,
        task_name=cfg.task_name,
        fine_tuning_only=cfg.fine_tuning_only,
        encoder_model_name=cfg.model.encoder_model_name,
        intermediate_training_config=intermediate_training_config,
        fine_tuning_config=fine_tuning_config,
        output_base_dir=output_base_dir,
        inter_early_stopping_patience=inter_patience,
        ft_early_stopping_patience=ft_patience,
    )

    main_hydra(data, config)

    config_save_path = os.path.join(output_base_dir, "resolved_config.yaml")
    OmegaConf.save(cfg, config_save_path)
    log.info("Resolved config saved to %s", config_save_path)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig):
    cite_main(cfg)


if __name__ == "__main__":
    hydra_entry()
