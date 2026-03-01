"""
training.py の Hydra 対応ラッパー。

training.py の main() を内部で呼び出しつつ、
output_base_dir が設定されている場合は Hydra 出力ディレクトリにログ・モデル・
評価結果を集約する。

training.py 自体は変更しないため、既存の *_run.py からの呼び出しには影響しない。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "..", "base"))

import torch
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
import pandas as pd

from model import CLS_bert, Bin_bert
from training import (
    Mydatasets,
    MyDatasetsInter,
    DataCollator,
    compute_metrics,
    load_tokenizer,
    make_pairs,
    check_config,
)
from url_cite_assets import SplitedData, TrainingConfig


def _resolve_paths(config: TrainingConfig) -> dict[str, str]:
    """output_base_dir が設定されている場合は Hydra 出力ディレクトリベースのパスを返す。
    設定されていない場合は従来のパス構成にフォールバックする。
    """
    if config.output_base_dir is not None:
        base = config.output_base_dir
        phase = "fine_tuning_only" if config.fine_tuning_only else "target"
        return {
            "inter_logging_dir": os.path.join(base, "intermediate", "logs"),
            "inter_output_dir": os.path.join(base, "intermediate", "model"),
            "inter_save_dir": os.path.join(base, "intermediate"),
            "ft_logging_dir": os.path.join(base, phase, "logs"),
            "ft_output_dir": os.path.join(base, phase, "model"),
            "ft_save_dir": os.path.join(base, phase),
            "results_dir": os.path.join(base, phase),
        }
    else:
        # Legacy path construction (same as training.py)
        legacy_base = f"./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}"
        phase = "fine_tuning_only" if config.fine_tuning_only else "target"
        return {
            "inter_logging_dir": f"./logs/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/intermediate",
            "inter_output_dir": f"{legacy_base}/intermediate/model",
            "inter_save_dir": f"{legacy_base}/intermediate",
            "ft_logging_dir": f"./logs/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/{phase}",
            "ft_output_dir": f"{legacy_base}/{phase}/model",
            "ft_save_dir": f"{legacy_base}/{phase}",
            "results_dir": f"{legacy_base}/{phase}",
        }


def main_hydra(
    data: SplitedData,
    config: TrainingConfig,
):
    """
    training.py の main() と同等の処理を行うが、
    出力パスの解決に _resolve_paths() を使い、
    early_stopping_patience を config から取得する。
    """

    # check config
    check_config(config)

    # check data size
    train_data_num = len(data["train_X"])
    valid_data_num = len(data["valid_X"])
    test_data_num = len(data["test_X"])
    total_data_num = train_data_num + valid_data_num + test_data_num

    print("original data size: ", total_data_num)
    print("train data size: ", train_data_num)
    print("valid data size: ", valid_data_num)
    print("test data size: ", test_data_num)

    # load tokenizer
    encoder_model_name = config.encoder_model_name
    tokenizer = load_tokenizer(encoder_model_name, config.special_tokens)

    # fix seed
    transformers.set_seed(config.training_seed)

    # resolve output paths
    paths = _resolve_paths(config)

    # ===== Intermediate task training =====
    bin_model: Bin_bert | None = None
    if not config.fine_tuning_only:
        inter_X: list[list[str]] = []
        inter_Ys: list[list[bool]] = []
        if config.n_sample:
            inter_X, inter_Ys = make_pairs(
                data["train_X"],
                data["train_labels"],
                config.n_sample,
                strategy="random",
            )
        elif config.sample_ratio:
            inter_X, inter_Ys = make_pairs(
                data["train_X"],
                data["train_labels"],
                int(config.sample_ratio * train_data_num * train_data_num),
            )
        train_df, valid_df = train_test_split(
            pd.DataFrame([inter_X] + inter_Ys).transpose(),
            test_size=0.2,
            random_state=config.inter_split_seed,
        )
        inter_train_X: list[list[str]] = train_df.iloc[:, 0].tolist()
        inter_train_Ys: list[list[bool]] = (
            train_df.iloc[:, 1:].transpose().values.tolist()
        )
        inter_valid_X: list[list[str]] = valid_df.iloc[:, 0].tolist()
        inter_valid_Ys: list[list[bool]] = (
            valid_df.iloc[:, 1:].transpose().values.tolist()
        )

        # create dataset
        inter_train_dataset = MyDatasetsInter(
            inter_train_X, inter_train_Ys, n_label=len(config.n_classes)
        )
        inter_valid_dataset = MyDatasetsInter(
            inter_valid_X, inter_valid_Ys, n_label=len(config.n_classes)
        )

        collator = DataCollator(tokenizer)

        # load model
        bin_model = Bin_bert(
            len(tokenizer), len(config.n_classes), model_name=encoder_model_name
        )
        bin_model.cuda()

        # training
        args = (
            config.intermediate_training_config
            if config.intermediate_training_config
            else {}
        )
        training_args = {
            "logging_dir": paths["inter_logging_dir"],
            "output_dir": paths["inter_output_dir"],
            "eval_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "fp16": True,
            "label_names": ["label"],
            "lr_scheduler_type": "constant",
            "metric_for_best_model": "loss",
            "load_best_model_at_end": True,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 100,
            "remove_unused_columns": False,
            "report_to": "all",
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-5,
            **args,
        }
        training_args = TrainingArguments(**training_args)

        trainer = Trainer(
            model=bin_model,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            args=training_args,
            train_dataset=inter_train_dataset,
            eval_dataset=inter_valid_dataset,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.inter_early_stopping_patience
                )
            ],
        )

        trainer.train(
            ignore_keys_for_eval=["input_ids", "attention_mask", "token_type_ids"]
        )

        # save model
        trainer.save_model(paths["inter_save_dir"])

    # ===== Target task (fine-tuning) =====
    train_dataset = Mydatasets(
        data["train_X"], data["train_labels"], n_label=len(config.n_classes)
    )
    valid_dataset = Mydatasets(
        data["valid_X"], data["valid_labels"], n_label=len(config.n_classes)
    )
    test_dataset = Mydatasets(
        data["test_X"], data["test_labels"], n_label=len(config.n_classes)
    )

    model = CLS_bert(
        len(tokenizer), config.n_classes, model_name=encoder_model_name
    ).cuda()
    if not config.fine_tuning_only and bin_model is not None:
        model.bert = bin_model.bert

    collator = DataCollator(tokenizer)

    args = config.fine_tuning_config if config.fine_tuning_config else {}
    training_args = {
        "logging_dir": paths["ft_logging_dir"],
        "output_dir": paths["ft_output_dir"],
        "eval_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "fp16": True,
        "label_names": ["label"],
        "lr_scheduler_type": "constant",
        "metric_for_best_model": "loss",
        "load_best_model_at_end": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 100,
        "remove_unused_columns": False,
        "report_to": "all",
        "learning_rate": 1e-5,
        **args,
    }
    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.ft_early_stopping_patience
            )
        ],
    )

    trainer.train(
        ignore_keys_for_eval=["input_ids", "attention_mask", "token_type_ids"]
    )

    # save model & trainer state
    trainer.save_model(paths["ft_save_dir"])
    trainer.save_state()

    # ===== Evaluation & save results =====
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    for split_name, dataset in [
        ("train", train_dataset),
        ("valid", valid_dataset),
        ("test", test_dataset),
    ]:
        pred = trainer.evaluate(dataset, ignore_keys=["loss"])
        result_path = os.path.join(results_dir, f"{split_name}_pred.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, indent=4, ensure_ascii=False)
        print(f"Saved {split_name} results to {result_path}")
