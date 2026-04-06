from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict


CITE_TOKEN = "[CITE]"
TEST_SEED = 111
TEST_SIZE = 0.1
TRAIN_SEED = 12539
TRAIN_SIZE = 1 / 9
DATA_PATH = "data/tsunokake_data_base.csv"

ROLE_MAP = {"Method": 0, "Material": 1, "補足資料": 2, "Mixed": 3}

TYPE_MAP = {
    "Dataset": 0,
    "Paper": 1,
    "Website": 2,
    "Tool": 3,
    "Code": 4,
    "Knowledge": 5,
    "DataSource": 6,
    "Mixed": 7,
    "Media": 8,
    "Document": 9,
}

FUNCTION_MAP = {
    "Extend（引用目的）": 0,
    "Other（引用目的）": 1,
    "Use（引用目的）": 2,
    "Introduce（引用目的）": 3,
    "Produce（引用目的）": 4,
    "Compare（引用目的）": 5,
}

SPECIAL_TOKENS = [CITE_TOKEN]


class SplitedData(TypedDict):
    train_X: list[str]
    train_labels: list[list[int]]
    valid_X: list[str]
    valid_labels: list[list[int]]
    test_X: list[str]
    test_labels: list[list[int]]


Strategy = Literal["random", "binary-balanced", "multi-balanced"]


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    mode: str = "online"
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_model: bool = False
    save_code: bool = False


@dataclass
class TrainingConfig:
    n_classes: list[int]
    special_tokens: list[str]
    training_seed: int
    task_name: str
    fine_tuning_only: bool = False
    inter_split_seed: Optional[int] = None
    n_sample: Optional[int] = None
    sample_ratio: Optional[float] = None
    pair_makeing_strategy: Strategy = "random"
    target_label_idx: int = 0
    encoder_model_name: str = "google/bert_uncased_L-12_H-768_A-12"
    intermediate_training_config: Optional[dict] = None
    fine_tuning_config: Optional[dict] = None
    output_base_dir: Optional[str] = None  # Hydra output directory (if None, use legacy paths)
    inter_early_stopping_patience: int = 2
    ft_early_stopping_patience: int = 5
    wandb: Optional[WandbConfig] = None
    resolved_config: Optional[dict] = None
