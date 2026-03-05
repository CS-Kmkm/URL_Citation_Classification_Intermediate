from datasets import load_dataset
from sklearn.model_selection import train_test_split

from training import TrainingConfig, main
from url_cite_assets import (
    SplitedData
)

TEST_PATH = "path/to/test_data"
TRAIN_PATH = "path/to/train_data"

LABEL_MAP = {
    0: 0,
    1: 1,
}


def load_data() -> SplitedData:
    test_dataset = load_dataset(
        "csv",
        data_files=TEST_PATH,
        delimiter="\t",
        column_names=["col", "label", "author", "text"]
    )
    train_dataset = load_dataset(
        "csv",
        data_files=TRAIN_PATH,
        delimiter="\t",
        column_names=["col", "label", "author", "text"],
        split="train"
    )

    # split train dataset into train and valid
    train_dataset = train_dataset.train_test_split(
        test_size=0.1, shuffle=True, seed=111)

    train_x = train_dataset['train']['text']
    train_y = list(
        map(lambda x: LABEL_MAP[x], train_dataset['train']['label']))
    valid_x = train_dataset['test']['text']
    valid_y = list(map(lambda x: LABEL_MAP[x], train_dataset['test']['label']))
    test_x = test_dataset['train']['text']
    test_y = list(map(lambda x: LABEL_MAP[x], test_dataset['train']['label']))

    return {
        'train_X': train_x,
        'train_labels': [train_y],
        'valid_X': valid_x,
        'valid_labels': [valid_y],
        'test_X': test_x,
        'test_labels': [test_y]
    }


def cola_main():
    data = load_data()

    assert len(data['train_X']) == len(data['train_labels'][0])
    assert len(data['valid_X']) == len(data['valid_labels'][0])
    assert len(data['test_X']) == len(data['test_labels'][0])

    n_classes = [len(LABEL_MAP)]

    share_config = {
        'n_classes': n_classes,
        'special_tokens': [],
        'n_sample': 100_000,
        # n_sample=1_000,
        'inter_split_seed': 111,
        # training_seed=8174,
        # 'training_seed':111,
        # 'training_seed':5374,
        'training_seed': 93279,
        # 'training_seed': 61843,
        # 'training_seed': 12345,
        'task_name': 'cola',
        'fine_tuning_only': True,
        'encoder_model_name': "bert-base-uncased",
        # encoder_model_name="bert-large-uncased",
        # "encoder_model_name": "roberta-base",
        'intermediate_training_config': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 16,
            'learning_rate': 1e-5,
            # 'per_device_train_batch_size': 2, # for large model
            # 'gradient_accumulation_steps': 4, # for large model
            # 'per_device_eval_batch_size': 2, # for large model
            # 'learning_rate': 4e-6, # for large model
        },
        'fine_tuning_config': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 16,
            # 'per_device_train_batch_size': 2, # for large model
            # 'gradient_accumulation_steps': 2, # for large model
            # 'per_device_eval_batch_size': 2,  # for large model
            # 'learning_rate': 4e-6, for large model
            'learning_rate': 1e-5,
        }
    }

    config = TrainingConfig(
        **share_config
    )

    print('start fine tuning only')

    main(
        data,
        config
    )

    share_config['fine_tuning_only'] = False

    config = TrainingConfig(
        **share_config
    )

    print('start intermediate training')

    main(
        data,
        config
    )


if __name__ == "__main__":
    cola_main()
