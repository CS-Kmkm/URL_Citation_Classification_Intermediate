import csv

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from training import TrainingConfig, main
from url_cite_assets import (
    SplitedData
)

FILE_PATH = "path/to/edos_labelled_aggregated.csv"

LABEL_MAP = {
    '3.4 condescending explanations or unwelcome advice': 0,
    '3.3 backhanded gendered compliments': 1,
    '1.2 incitement and encouragement of harm': 2,
    '2.1 descriptive attacks': 3,
    '2.3 dehumanising attacks & overt sexual objectification': 4,
    '4.1 supporting mistreatment of individual women': 5,
    '3.1 casual use of gendered slurs, profanities, and insults': 6,
    '4.2 supporting systemic discrimination against women as a group': 7,
    '2.2 aggressive and emotive attacks': 8,
    '3.2 immutable gender differences and gender stereotypes': 9,
    '1.1 threats of harm': 10
}


def load_data() -> SplitedData:
    data = list(csv.reader(open(FILE_PATH, encoding='utf-8')))
    data = list(filter(lambda x: x[2] == 'sexist', data))

    train_dataset = dict()
    train_data = list(filter(lambda x: x[5] == 'train', data))
    train_dataset['text'] = [d[1] for d in train_data]
    train_dataset['label'] = [d[4] for d in train_data]

    valid_dataset = dict()
    valid_data = list(filter(lambda x: x[5] == 'dev', data))
    valid_dataset['text'] = [d[1] for d in valid_data]
    valid_dataset['label'] = [d[4] for d in valid_data]

    test_dataset = dict()
    test_data = list(filter(lambda x: x[5] == 'test', data))
    test_dataset['text'] = [d[1] for d in test_data]
    test_dataset['label'] = [d[4] for d in test_data]

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


def edos_task3_main():
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
        'task_name': 'edos-task3',
        'fine_tuning_only': True,
        'encoder_model_name': "bert-base-uncased",
        # encoder_model_name="bert-large-uncased",
        # encoder_model_name="roberta-base",
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
    redos_task3_main()
