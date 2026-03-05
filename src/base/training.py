# training script for the model with trainer
from collections import Counter
import json
import random
import warnings

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import transformers
from transformers import EvalPrediction, TrainingArguments, Trainer, EarlyStoppingCallback  # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import CLS_bert, Bin_bert
from url_cite_assets import SplitedData, TrainingConfig


class Mydatasets(Dataset):
    def __init__(self, X: list[str], Y: list[list[int]], n_label=1) -> None:
        self.X = X
        self.Y = Y  # (n_label, datanum)
        self.n_label = n_label

        self.datanum = len(self.X)

    def __len__(self) -> int:
        return self.datanum

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        return self.X[idx], [self.Y[i][idx] for i in range(self.n_label)]


class MyDatasetsInter(Dataset):
    def __init__(self, X: list[list[str]], Y: list[list[bool]], n_label=1) -> None:
        self.X = X  # (data_num, 2)
        self.Y = Y  # (n_label, datanum)
        self.n_label = n_label

        self.datanum = len(self.X)

    def __len__(self) -> int:
        return self.datanum

    def __getitem__(self, idx: int) -> tuple[list[str], list[bool]]:
        return self.X[idx], [self.Y[i][idx] for i in range(self.n_label)]


class DataCollator():
    def __init__(self, tokenizer, max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        x, y = list(zip(*batch))
        n_label = len(y[-1])
        labels = [torch.tensor([i[j] for i in y]).long()
                  for j in range(n_label)]

        inputs = self.tokenizer(
            x,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs['label'] = labels

        return inputs


def compute_metrics(res: EvalPrediction):
    n_label = len(res.label_ids)
    labels = res.label_ids
    # check if the classification is binary or not
    if res.predictions[0].ndim == 1:
        preds = [res.predictions[i] > 0.5 for i in range(n_label)]
        accs = [accuracy_score(labels[i], preds[i]) for i in range(n_label)]
        recalls = [recall_score(labels[i], preds[i], average='binary')
                   for i in range(n_label)]
        precisions = [precision_score(
            labels[i], preds[i], average='binary') for i in range(n_label)]
        f1s = [f1_score(labels[i], preds[i], average='binary')
               for i in range(n_label)]
        avr_acc = np.mean(accs)  # type: ignore
        avr_recall = np.mean(recalls)  # type: ignore
        avr_precision = np.mean(precisions)  # type: ignore
        avr_f1 = np.mean(f1s)  # type: ignore
        ret = {f'label{i}_acc': accs[i] for i in range(n_label)}
        ret.update({f'label{i}_recall': recalls[i] for i in range(n_label)})
        ret.update(
            {f'label{i}_precision': precisions[i] for i in range(n_label)})
        ret.update({f'label{i}_f1': f1s[i] for i in range(n_label)})
        ret.update({
            'avr_acc': avr_acc,
            'avr_recall': avr_recall,
            'avr_precision': avr_precision,
            'avr_f1': avr_f1
        })
        return ret

    preds = [res.predictions[i].argmax(axis=1) for i in range(n_label)]

    accs = [accuracy_score(labels[i], preds[i]) for i in range(n_label)]
    macro_recalls = [recall_score(
        labels[i], preds[i], average='macro') for i in range(n_label)]
    micro_recalls = [recall_score(
        labels[i], preds[i], average='micro') for i in range(n_label)]
    macro_precisions = [precision_score(
        labels[i], preds[i], average='macro') for i in range(n_label)]
    micro_precisions = [precision_score(
        labels[i], preds[i], average='micro') for i in range(n_label)]
    macro_f1s = [f1_score(labels[i], preds[i], average='macro')
                 for i in range(n_label)]
    micro_f1s = [f1_score(labels[i], preds[i], average='micro')
                 for i in range(n_label)]
    weighted_f1s = [f1_score(labels[i], preds[i], average='weighted')
                    for i in range(n_label)]

    avr_acc = np.mean(accs)  # type: ignore
    avr_macro_recall = np.mean(macro_recalls)  # type: ignore
    avr_micro_recall = np.mean(micro_recalls)  # type: ignore
    avr_macro_precision = np.mean(macro_precisions)  # type: ignore
    avr_micro_precision = np.mean(micro_precisions)  # type: ignore
    avr_macro_f1 = np.mean(macro_f1s)  # type: ignore
    avr_micro_f1 = np.mean(micro_f1s)  # type: ignore

    ret = {f'label{i}_acc': accs[i] for i in range(n_label)}
    ret.update(
        {f'label{i}_macro_recall': macro_recalls[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_micro_recall': micro_recalls[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_macro_precision': macro_precisions[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_micro_precision': micro_precisions[i] for i in range(n_label)})
    ret.update({f'label{i}_macro_f1': macro_f1s[i] for i in range(n_label)})
    ret.update({f'label{i}_micro_f1': micro_f1s[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_weighted_f1': weighted_f1s[i] for i in range(n_label)})
    ret.update({'avr_acc': avr_acc,
                'avr_macro_recall': avr_macro_recall,
                'avr_micro_recall': avr_micro_recall,
                'avr_macro_precision': avr_macro_precision,
                'avr_micro_precision': avr_micro_precision,
                'avr_macro_f1': avr_macro_f1,
                'avr_micro_f1': avr_micro_f1
                })

    return ret


def load_tokenizer(model_name: str, special_tokens: list[str]):
    if 'deberta' in model_name:
        tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    return tokenizer


def make_pairs(
    data: list[str],
    labels: list[list[int]],
    n_samples: int,
    strategy: str = 'random',
    target_label_idx: int = 0  # necessary if strategy is 'binary_balanced'
) -> tuple[list[list[str]], list[list[bool]]]:

    if strategy == 'random':
        n_label = len(labels)
        X: list[list[str]] = []
        Ys: list[list[bool]] = [list() for _ in range(n_label)]
        for i in range(n_samples):
            idx1 = random.randint(0, len(data)-1)
            idx2 = random.randint(0, len(data)-1)

            X.append([data[idx1], data[idx2]])
            for j in range(n_label):
                Ys[j].append(labels[j][idx1] == labels[j][idx2])

        return X, Ys
    elif strategy == 'binary_balanced':
        n_label = len(labels)
        concat_labels = [[j, ''.join(
            [str(labels[i][j]) for i in range(n_label)])] for j in range(len(data))]
        print(len(concat_labels))
        # concat_labels = [[i, str(labels[i][target_label_idx])] for i in range(len(labels))]
        X: list[list[str]] = []
        Ys: list[list[bool]] = [list() for _ in range(n_label)]
        for i in range(int(n_samples / 2)):
            print(f'{i}/{int(n_samples / 2)}', end='\r')
            choices = list()
            while len(choices) == 0:
                idx1 = random.randint(0, len(data)-1)
                choices = list(
                    filter(lambda x: x[1] != concat_labels[idx1][1], concat_labels))
                choices = [i[0] for i in choices]
                # print(choices)
            idx2 = random.choice(choices)

            X.append([data[idx1], data[idx2]])
            for j in range(n_label):
                Ys[j].append(labels[j][idx1] == labels[j][idx2])

            choices = list()
            while len(choices) == 0:
                idx1 = random.randint(0, len(data)-1)
                choices = list(
                    filter(lambda x: x[1] == concat_labels[idx1][1], concat_labels))
                choices = [i[0] for i in choices]
            idx2 = random.choice(choices)

            X.append([data[idx1], data[idx2]])
            for j in range(n_label):
                Ys[j].append(labels[j][idx1] == labels[j][idx2])

        print([Counter(Ys[i]) for i in range(n_label)])

        return X, Ys


def check_config(config: TrainingConfig):
    # ignore below if fine_tuning_only is True
    if not config.fine_tuning_only:
        return
    # check n_sample or sample_ratio is set
    if config.n_sample is None and config.sample_ratio is None:
        raise ValueError('n_sample or sample_ratio must be set')
    # check n_sample and sample_ratio is not set at the same time (warning, n_sample is used)
    elif config.n_sample is not None and config.sample_ratio is not None:
        warnings.warn(
            'n_sample and sample_ratio is set at the same time, n_sample is used')
    # check inter_split_seed is set
    if config.inter_split_seed is None:
        raise ValueError('inter_split_seed must be set')

# def main(
#         data:SplitedData,
#         n_classes:list[int],
#         special_tokens:list[str],
#         n_sample:int,
#         inter_split_seed:int,
#         training_seed:int,
#         task_name:str,
#         fine_tuning_only:bool = False,
#     ):


def main(
    data: SplitedData,
    config: TrainingConfig,
):

    # transformers.logging.set_verbosity_error()

    # check config
    check_config(config)

    # check data size
    train_data_num = len(data['train_X'])
    valid_data_num = len(data['valid_X'])
    test_data_num = len(data['test_X'])
    total_data_num = train_data_num + valid_data_num + test_data_num

    print('original data size: ', total_data_num)
    print('train data size: ', train_data_num)
    print('valid data size: ', valid_data_num)
    print('test data size: ', test_data_num)

    # load tokenizer
    encoder_model_name = config.encoder_model_name
    tokenizer = load_tokenizer(encoder_model_name, config.special_tokens)

    # fix seed
    transformers.set_seed(config.training_seed)

    # training on intermediate task (if fine_tuning_only is False)
    # make pairs
    bin_model: Bin_bert | None = None
    if not config.fine_tuning_only:
        inter_X: list[list[str]] = []
        inter_Ys: list[list[bool]] = []
        if config.n_sample:
            inter_X, inter_Ys = make_pairs(
                data['train_X'], data['train_labels'], config.n_sample, strategy='random')
        elif config.sample_ratio:
            inter_X, inter_Ys = make_pairs(data['train_X'], data['train_labels'], int(
                config.sample_ratio * train_data_num * train_data_num))
        train_df, valid_df = train_test_split(
            pd.DataFrame([inter_X] + inter_Ys).transpose(),
            test_size=0.2,
            random_state=config.inter_split_seed
        )
        inter_train_X: list[list[str]] = train_df.iloc[:, 0].tolist()
        inter_train_Ys: list[list[bool]] = train_df.iloc[:,
                                                         1:].transpose().values.tolist()
        inter_valid_X: list[list[str]] = valid_df.iloc[:, 0].tolist()
        inter_valid_Ys: list[list[bool]] = valid_df.iloc[:,
                                                         1:].transpose().values.tolist()

        # create dataset
        inter_train_dataset = MyDatasetsInter(
            inter_train_X,
            inter_train_Ys,
            n_label=len(config.n_classes)
        )
        inter_valid_dataset = MyDatasetsInter(
            inter_valid_X,
            inter_valid_Ys,
            n_label=len(config.n_classes)
        )

        collator = DataCollator(tokenizer)

        # load model
        bin_model = Bin_bert(
            len(tokenizer),
            len(config.n_classes),
            model_name=encoder_model_name
        )
        bin_model.cuda()

        # training
        args = config.intermediate_training_config if config.intermediate_training_config else {}
        training_args = {
            'logging_dir': f'./logs/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/intermediate',
            'output_dir': f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/intermediate/model',
            'evaluation_strategy': 'epoch',
            'logging_strategy': 'epoch',
            'save_strategy': 'epoch',
            'save_total_limit': 1,
            'overwrite_output_dir': True,
            'fp16': True,
            'label_names': ['label'],
            'lr_scheduler_type': 'constant',
            # lr_scheduler_type='cosine',
            # metric_for_best_model='avr_f1',
            'metric_for_best_model': 'loss',
            'load_best_model_at_end': True,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 16,
            # per_device_train_batch_size=32,
            # per_device_eval_batch_size=64,
            # warmup_steps=1000,
            'num_train_epochs': 100,
            # 'num_train_epochs': 10,
            'remove_unused_columns': False,
            'report_to': 'all',
            'gradient_accumulation_steps': 4,
            'learning_rate': 1e-5,
            **args
        }
        training_args = TrainingArguments(
            **training_args
        )

        trainer = Trainer(
            model=bin_model,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            args=training_args,
            train_dataset=inter_train_dataset,
            eval_dataset=inter_valid_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train(ignore_keys_for_eval=[
                      'input_ids', 'attention_mask', 'token_type_ids'])

        # save model
        out_dir = f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/intermediate'
        trainer.save_model(out_dir)

    # training on target task
    # create dataset
    train_dataset = Mydatasets(
        data['train_X'],
        data['train_labels'],
        n_label=len(config.n_classes)
    )
    valid_dataset = Mydatasets(
        data['valid_X'],
        data['valid_labels'],
        n_label=len(config.n_classes)
    )
    test_dataset = Mydatasets(
        data['test_X'],
        data['test_labels'],
        n_label=len(config.n_classes)
    )

    model = CLS_bert(
        len(tokenizer),
        config.n_classes,
        model_name=encoder_model_name
    ).cuda()
    if not config.fine_tuning_only and bin_model is not None:
        model.bert = bin_model.bert
    # model = CLS_bert_layernorm(len(tokenizer), model_name=model_name).cuda()
    # model.load_state_dict(torch.load('./output/model/pytorch_model.bin'))
    # model.load_state_dict(
    #     torch.load('../pretraining/balanced_bert_base_add_feature_by_epoch/res/111/300000/12539/pretrained_bert.pth'),
    #     strict=False
    # )

    logging_dir = f'./logs/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only' if config.fine_tuning_only else f'./logs/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target'
    output_dir = f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only/model' if config.fine_tuning_only else f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target/model'

    collator = DataCollator(tokenizer)

    args = config.fine_tuning_config if config.fine_tuning_config else {}
    training_args = {
        'logging_dir': logging_dir,
        'output_dir': output_dir,
        'evaluation_strategy': 'epoch',
        'logging_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 1,
        'overwrite_output_dir': True,
        'fp16': True,
        'label_names': ['label'],
        'lr_scheduler_type': 'constant',
        # 'lr_scheduler_type': 'cosine',
        # 'metric_for_best_model': 'avr_macro_f1',
        'metric_for_best_model': 'loss',
        'load_best_model_at_end': True,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 16,
        # 'per_device_train_batch_size': 32,
        # 'per_device_eval_batch_size': 64,
        # 'warmup_steps': 1000,
        'num_train_epochs': 100,
        'remove_unused_columns': False,
        'report_to': 'all',
        'learning_rate': 1e-5,
        **args
    }
    training_args = TrainingArguments(
        **training_args
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train(ignore_keys_for_eval=[
                  'input_ids', 'attention_mask', 'token_type_ids'])

    # save model
    out_dir = f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only' if config.fine_tuning_only else f'./output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target'
    trainer.save_model(out_dir)
    # save trainer
    trainer.save_state()

    # evaluation (train)
    train_pred = trainer.evaluate(train_dataset, ignore_keys=['loss'])
    if config.fine_tuning_only:
        json.dump(train_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only/train_pred.json', 'w'), indent=4, ensure_ascii=False)
    else:
        json.dump(train_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target/train_pred.json', 'w'), indent=4, ensure_ascii=False)

    # evaluation (valid)
    valid_pred = trainer.evaluate(valid_dataset, ignore_keys=['loss'])
    if config.fine_tuning_only:
        json.dump(valid_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only/valid_pred.json', 'w'), indent=4, ensure_ascii=False)
    else:
        json.dump(valid_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target/valid_pred.json', 'w'), indent=4, ensure_ascii=False)

    # evaluation (test)
    test_pred = trainer.evaluate(test_dataset, ignore_keys=['loss'])

    if config.fine_tuning_only:
        json.dump(test_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/fine_tuning_only/test_pred.json', 'w'), indent=4, ensure_ascii=False)
    else:
        json.dump(test_pred, open(
            f'output/{config.task_name}/{config.encoder_model_name}/{config.training_seed}/target/test_pred.json', 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # main()
    pass
