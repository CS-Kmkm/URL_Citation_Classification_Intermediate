import os
import re
import math

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from training import main
from url_cite_assets import (
    CITE_TOKEN, 
    TEST_SEED, 
    TEST_SIZE, 
    TRAIN_SEED, 
    TRAIN_SIZE, 
    DATA_PATH, 
    ROLE_MAP, 
    TYPE_MAP, 
    FUNCTION_MAP, 
    SPECIAL_TOKENS,
    SplitedData,
    TrainingConfig
)

def replace_tag(sentenses: pd.Series) -> list[str]:
    # replace [Cite_***] to [CITE] token
    rule = re.compile(r'\[Cite[^\[\] ]*\]')
    sentense_replaced:list[str] = list()
    for sentense in sentenses:
        sentense_replaced.append(rule.sub(CITE_TOKEN, sentense))

    return sentense_replaced

def get_3sent(paragraphs: list[str]) -> list[list[str]]:
    ret:list[list[str]] = list()
    for paragraph in paragraphs:
        sentenses: list[str] = nltk.sent_tokenize(paragraph)
        if not len(sentenses):
            print("!!!")
        if len(sentenses) < 4:
            ret.append(sentenses)
            continue
        else:
            for i in range(len(sentenses)):
                if CITE_TOKEN in sentenses[i]:
                    if i == 0:
                        ret.append(sentenses[i:i+2])
                    elif i == len(sentenses)-1:
                        ret.append(sentenses[i-1:i+1])
                    else:
                        ret.append(sentenses[i-1:i+2])
                    break
                if i == len(sentenses)-1:
                    print(sentenses)
    return ret

def preprocess(data:pd.DataFrame, test_df:pd.DataFrame, seed:int):
    # split data to train and valid
    train_df, valid_df = train_test_split(
        data, shuffle=True, 
        random_state=seed, 
        test_size=1/9
    )

    # replace [Cite_***] to [CITE] token
    train_X_paragraph = replace_tag(train_df['citation-paragraph'])
    valid_X_paragraph = replace_tag(valid_df['citation-paragraph'])
    test_X_paragraph = replace_tag(test_df['citation-paragraph'])

    # get 3 sentenses (include citation sentense)
    train_X_sents = get_3sent(train_X_paragraph)
    valid_X_sents = get_3sent(valid_X_paragraph)
    test_X_sents = get_3sent(test_X_paragraph)
    
    # concat sentenses
    train_X = [' '.join(i) for i in train_X_sents]
    valid_X = [' '.join(i) for i in valid_X_sents]
    test_X = [' '.join(i) for i in test_X_sents]
    
    # get section title (use most deep section title)
    train_title = list(
        map(
            lambda x:eval(x.replace(r'\'', '"'))[-1], 
            train_df['passage-title'].tolist()
        )
    )
    valid_title = list(
        map(
            lambda x:eval(x.replace(r'\'', '"'))[-1], 
            valid_df['passage-title'].tolist()
        )
    )
    test_title = list(
        map(
            lambda x:eval(x.replace(r'\'', '"'))[-1], 
            test_df['passage-title'].tolist()
        )
    )

    # get citation info (reference text, footnote text or empty)
    train_info = train_df['citation-info'].tolist()
    valid_info = valid_df['citation-info'].tolist()
    test_info = test_df['citation-info'].tolist()

    train_info = list(
        map(
            lambda x: '' if type(x) == type(float()) and math.isnan(x) else x, 
            train_info
        )
    )
    valid_info = list(
        map(
            lambda x: '' if type(x) == type(float()) and math.isnan(x) else x, 
            valid_info
        )
    )
    test_info = list(
        map(
            lambda x: '' if type(x) == type(float()) and math.isnan(x) else x, 
            test_info
        )
    )

    # concat title, sentense and info with [SEP] token
    train_X:list[str] = [title+'[SEP]'+sentense+'[SEP]'+info for title, sentense, info in zip(train_title, train_X, train_info)]
    valid_X:list[str] = [title+'[SEP]'+sentense+'[SEP]'+info for title, sentense, info in zip(valid_title, valid_X, valid_info)]
    test_X:list[str] = [title+'[SEP]'+sentense+'[SEP]'+info for title, sentense, info in zip(test_title, test_X, test_info)]

    # map class name(str) to number
    train_role_y = list(
        map(
            lambda x:ROLE_MAP[x], 
            train_df['role'].tolist()
        )
    )
    train_type_y = list(
        map(
            lambda x:TYPE_MAP[x], 
            train_df['type'].tolist()
        )
    )
    train_function_y = list(
        map(
            lambda x:FUNCTION_MAP[x], 
            train_df['function'].tolist()
        )
    )

    valid_role_y = list(
        map(
            lambda x:ROLE_MAP[x], 
            valid_df['role'].tolist()
        )
    )
    valid_type_y = list(
        map(
            lambda x:TYPE_MAP[x], 
            valid_df['type'].tolist()
        )
    )
    valid_function_y = list(
        map(
            lambda x:FUNCTION_MAP[x], 
            valid_df['function'].tolist()
        )
    )

    test_role_y = list(
        map(
            lambda x:ROLE_MAP[x], 
            test_df['role'].tolist()
        )
    )
    test_type_y = list(
        map(
            lambda x:TYPE_MAP[x], 
            test_df['type'].tolist()
        )
    )
    test_function_y = list(
        map(
            lambda x:FUNCTION_MAP[x], 
            test_df['function'].tolist()
        )
    )

    return (
        train_X, train_role_y, train_type_y, train_function_y,
        valid_X, valid_role_y, valid_type_y, valid_function_y,
        test_X, test_role_y, test_type_y, test_function_y
    )

def load_data() -> SplitedData:
    ''' required function
    1. load data
    2. split data to train, valid and test (if needed, 8:1:1)
    '''
    df = pd.read_csv(
        DATA_PATH,
        encoding='utf-8',
        index_col=0
    )

    train_df:pd.DataFrame
    test_df:pd.DataFrame
    train_df, test_df = train_test_split(
        df, 
        shuffle=True, 
        random_state=TEST_SEED, 
        test_size=TEST_SIZE
    )

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    preprocessed_data = preprocess(
        train_df, 
        test_df, 
        TRAIN_SEED
    )

    return {
        'train_X': preprocessed_data[0],
        'train_labels': list(preprocessed_data[1:4]),
        'valid_X': preprocessed_data[4],
        'valid_labels': list(preprocessed_data[5:8]),
        'test_X': preprocessed_data[8],
        'test_labels': list(preprocessed_data[9:12])
    }

def cite_main(n_sample=300_000, seed=111, fine_tune_only=False):
    # load data
    data = load_data()

    n_classes = [4, 10, 6]

    config = TrainingConfig(
        n_classes,
        special_tokens=SPECIAL_TOKENS,
        n_sample=n_sample,
        # n_sample=1_000,
        inter_split_seed=111,
        # training_seed=8174,
        # training_seed=111,
        # training_seed=5374,
        # training_seed=93279,
        training_seed=seed,
        task_name=f'url_citation_random_{n_sample}',
        fine_tuning_only=fine_tune_only,
        encoder_model_name="bert-base-uncased",
        # encoder_model_name="bert-large-uncased",
        # encoder_model_name="roberta-base",
        intermediate_training_config={
            'per_device_train_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 16,
            # 'per_device_train_batch_size': 4,
            # 'gradient_accumulation_steps': 2,
            # 'per_device_eval_batch_size': 8,
            # 'per_device_train_batch_size': 2, # for large model
            # 'gradient_accumulation_steps': 4, # for large model
            # 'per_device_eval_batch_size': 2, # for large model
            # 'learning_rate': 4e-6, # for large model
            'learning_rate': 1e-5,
            #'evaluation_strategy': 'steps',
            #'eval_steps': 2000,
            #'save_strategy': 'steps',
            #'save_steps': 2000,
        },
        fine_tuning_config={
            'per_device_train_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'per_device_eval_batch_size': 16,
            # 'per_device_train_batch_size': 8,
            # 'gradient_accumulation_steps': 2,
            # 'per_device_eval_batch_size': 16,
            # 'per_device_train_batch_size': 2, # for large model
            # 'gradient_accumulation_steps': 2, # for large model
            # 'per_device_eval_batch_size': 2,  # for large model
            # 'learning_rate': 4e-6, for large model
            'learning_rate': 1e-5,
        }
    )

    main(
        data,
        config
    )

if __name__ == '__main__':
    data_num_range = list(range(1_000, 5_000, 1_000)) + list(range(5_000, 100_000, 5_000))
    # seeds = [111, 5374, 93279]
    seeds = [7429, 429834]    
    for seed in seeds:
        for data_num in data_num_range:
            print(f"start {data_num} {seed}")
            if os.path.exists(f'./output//url_citation_random_{data_num}/bert-base-uncased/{seed}/target/test_pred.json'):
                print(f'./output/url_citation_random_{data_num}/pytorch_model.bin exists')
                continue
            cite_main(data_num, seed, False)
            # cite_main(data_num, seed, True)
