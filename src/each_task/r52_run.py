from datasets import load_dataset
from sklearn.model_selection import train_test_split

from training import TrainingConfig, main
from url_cite_assets import (
    SplitedData
)

TEST_PATH = "path/to/r52-test-all-terms.txt"
TRAIN_PATH = "path/to/r52-train-all-terms.txt"

labels = ['gas', 'iron-steel', 'nat-gas', 'grain', 'orange', 'gold', 'rubber', 'nickel', 'copper', 'housing', 'cpi', 'cocoa', 'strategic-metal', 'sugar', 'coffee', 'crude', 'jet', 'pet-chem', 'instal-debt', 'tin', 'acq', 'jobs', 'lumber', 'earn', 'fuel',
          'platinum', 'zinc', 'money-fx', 'money-supply', 'lei', 'cpu', 'retail', 'heat', 'carcass', 'veg-oil', 'trade', 'dlr', 'ship', 'gnp', 'alum', 'cotton', 'ipi', 'lead', 'reserves', 'meal-feed', 'wpi', 'potato', 'income', 'bop', 'livestock', 'interest']
LABEL_MAP = {
    'gas': 0,
    'iron-steel': 1,
    'nat-gas': 2,
    'grain': 3,
    'orange': 4,
    'gold': 5,
    'rubber': 6,
    'nickel': 7,
    'copper': 8,
    'housing': 9,
    'cpi': 10,
    'cocoa': 11,
    'strategic-metal': 12,
    'sugar': 13,
    'coffee': 14,
    'crude': 15,
    'jet': 16,
    'pet-chem': 17,
    'instal-debt': 18,
    'tin': 19,
    'acq': 20,
    'jobs': 21,
    'lumber': 22,
    'earn': 23,
    'fuel': 24,
    'platinum': 25,
    'zinc': 26,
    'money-fx': 27,
    'money-supply': 28,
    'lei': 29,
    'cpu': 30,
    'retail': 31,
    'heat': 32,
    'carcass': 33,
    'veg-oil': 34,
    'trade': 35,
    'dlr': 36,
    'ship': 37,
    'gnp': 38,
    'alum': 39,
    'cotton': 40,
    'ipi': 41,
    'lead': 42,
    'reserves': 43,
    'meal-feed': 44,
    'wpi': 45,
    'potato': 46,
    'income': 47,
    'bop': 48,
    'livestock': 49,
    'interest': 50,
    'tea': 51
}


def load_data() -> SplitedData:
    test_dataset = load_dataset(
        "csv",
        data_files=TEST_PATH,
        delimiter="\t",
        column_names=["label", "text"]
    )
    train_dataset = load_dataset(
        "csv",
        data_files=TRAIN_PATH,
        delimiter="\t",
        column_names=["label", "text"],
        split="train[:90%]"
    )
    valid_dataset = load_dataset(
        "csv",
        data_files=TRAIN_PATH,
        delimiter="\t",
        column_names=["label", "text"],
        split="train[90%:]"
    )

    train_x = train_dataset['text']
    train_y = list(map(lambda x: LABEL_MAP[x], train_dataset['label']))
    valid_x = valid_dataset['text']
    valid_y = list(map(lambda x: LABEL_MAP[x], valid_dataset['label']))
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


def r52_main():
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
        # 'training_seed': 93279,
        # 'training_seed': 12345,
        'training_seed': 61843,
        'task_name': 'r52',
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
    r52_main()
