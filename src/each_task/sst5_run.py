from datasets import load_dataset
from sklearn.model_selection import train_test_split

from training import TrainingConfig, main
from url_cite_assets import (
    SplitedData
)


LABEL_MAP_NUM = 5
LABEL_MAP = {i: i for i in range(LABEL_MAP_NUM)}


def load_data() -> SplitedData:
    shuffle_seed = 111
    dataset_name = "SetFit/sst5"
    dataset = load_dataset(dataset_name)
    test_dataset = dataset['test']
    valid_dataset = dataset['validation']
    train_dataset = dataset['train']

    train_x = train_dataset['text']
    train_y = list(map(lambda x: LABEL_MAP[x], train_dataset['label']))
    valid_x = valid_dataset['text']
    valid_y = list(map(lambda x: LABEL_MAP[x], valid_dataset['label']))
    test_x = test_dataset['text']
    test_y = list(map(lambda x: LABEL_MAP[x], test_dataset['label']))

    return {
        'train_X': train_x,
        'train_labels': [train_y],
        'valid_X': valid_x,
        'valid_labels': [valid_y],
        'test_X': test_x,
        'test_labels': [test_y]
    }


def sst5_main():
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
        # 'training_seed':53748,
        'training_seed': 93279,
        # 'training_seed': 12345,
        # 'training_seed': 61843,
        'task_name': 'sst5',
        'fine_tuning_only': True,
        'encoder_model_name': "bert-base-uncased",
        # encoder_model_name="bert-large-uncased",
        # encoder_model_name="roberta-base",
        'intermediate_training_config': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 16,
            'learning_rate': 1e-5,
            # 'num_train_epochs': 5,
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
    sst5_main()
