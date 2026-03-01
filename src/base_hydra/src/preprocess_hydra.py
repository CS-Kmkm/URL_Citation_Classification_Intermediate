"""
preprocess_hydra.py — モデル対応の前処理モジュール。

wada_src/url_cite_run.py の preprocess() との差分:
- [CITE] トークンをモデルごとに適切な形式に変換する。
  - BERT/RoBERTa 系: add_tokens() で追加したトークン [CITE] をそのまま使用
  - ModernBERT 系:   add_tokens() で追加した [CITE] をそのまま使用（同じ）
  ※ add_tokens() による埋め込み拡張は training_hydra.py 内の load_tokenizer() で行うため、
    前処理側で文字列を変換する必要はない。

- テキスト連結時のセパレータをモデルの separator token に合わせる。
  - BERT 系     : "[SEP]" (tokenizer が separator special token として認識する)
  - RoBERTa 系  : " </s> " (RoBERTa の separator token)
  - ModernBERT 系: " </s> " (ModernBERT の separator token)

Note:
  BERT 系では文字列中に "[SEP]" を埋め込んでも AutoTokenizer が separator token として
  解釈するため、従来の wada_src の挙動と互換性がある。
  一方 ModernBERT / RoBERTa は "[SEP]" を未知語として処理してしまうため、
  モデルの実際の separator token を使う必要がある。
"""

import math
import re

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

# wada_src の定数を再利用
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "wada_src"))

from url_cite_assets import (
    CITE_TOKEN,
    ROLE_MAP,
    TYPE_MAP,
    FUNCTION_MAP,
)


def _is_modernbert(model_name: str) -> bool:
    return "modernbert" in model_name.lower()


def _is_roberta(model_name: str) -> bool:
    return "roberta" in model_name.lower()


def get_sep_token(model_name: str) -> str:
    """モデルに応じたセパレータ文字列を返す。

    文字列レベルでセパレータを埋め込む際に使用する。
    トークナイザーがその文字列を special token として認識するかは保証されないが、
    少なくともモデルの実際の separator token 文字列を使うことで
    BPE 分割を避け、vocab に登録済みの token として処理される可能性を高める。

    - BERT 系:              "[SEP]"  — vocab 登録済み special token (ID=102)
    - RoBERTa / ModernBERT: "</s>"   — vocab 登録済み separator token
    """
    if _is_modernbert(model_name) or _is_roberta(model_name):
        return "</s>"
    return "[SEP]"


def replace_tag(sentences: pd.Series) -> list[str]:
    """[Cite_***] を [CITE] トークンに統一する（wada_src と同一ロジック）。"""
    rule = re.compile(r"\[Cite[^\[\] ]*\]")
    return [rule.sub(CITE_TOKEN, s) for s in sentences]


def get_3sent(paragraphs: list[str]) -> list[list[str]]:
    """引用文を含む前後3文を抽出する（wada_src と同一ロジック）。"""
    ret: list[list[str]] = []
    for paragraph in paragraphs:
        sentences: list[str] = nltk.sent_tokenize(paragraph)
        if not sentences:
            print("!!!")
        if len(sentences) < 4:
            ret.append(sentences)
            continue
        for i, sent in enumerate(sentences):
            if CITE_TOKEN in sent:
                if i == 0:
                    ret.append(sentences[i : i + 2])
                elif i == len(sentences) - 1:
                    ret.append(sentences[i - 1 : i + 1])
                else:
                    ret.append(sentences[i - 1 : i + 2])
                break
            if i == len(sentences) - 1:
                print(sentences)
    return ret


def preprocess(
    data: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    model_name: str = "bert-base-uncased",
):
    """前処理を実行し、モデルに応じたセパレータでテキストを連結する。

    Args:
        data: 学習・検証に使うデータフレーム（test_df を除いたもの）
        test_df: テストデータフレーム
        seed: train/valid 分割のランダムシード
        model_name: 使用するモデル名。セパレータの選択に使用。

    Returns:
        wada_src の preprocess() と同一の 12 要素タプル:
        (train_X, train_role_y, train_type_y, train_func_y,
         valid_X, valid_role_y, valid_type_y, valid_func_y,
         test_X,  test_role_y,  test_type_y,  test_func_y)
    """
    sep = get_sep_token(model_name)

    # train / valid 分割
    train_df, valid_df = train_test_split(data, shuffle=True, random_state=seed, test_size=1 / 9)

    # [Cite_***] → [CITE]
    train_X_paragraph = replace_tag(train_df["citation-paragraph"])
    valid_X_paragraph = replace_tag(valid_df["citation-paragraph"])
    test_X_paragraph = replace_tag(test_df["citation-paragraph"])

    # 引用文を含む前後3文を抽出
    train_X_sents = get_3sent(train_X_paragraph)
    valid_X_sents = get_3sent(valid_X_paragraph)
    test_X_sents = get_3sent(test_X_paragraph)

    # 文を連結
    train_X = [" ".join(s) for s in train_X_sents]
    valid_X = [" ".join(s) for s in valid_X_sents]
    test_X = [" ".join(s) for s in test_X_sents]

    # セクションタイトル（最深階層）
    def extract_last_title(series: pd.Series) -> list[str]:
        return list(map(lambda x: eval(x.replace(r"\'", '"'))[-1], series.tolist()))

    train_title = extract_last_title(train_df["passage-title"])
    valid_title = extract_last_title(valid_df["passage-title"])
    test_title = extract_last_title(test_df["passage-title"])

    # 参考文献情報（NaN は空文字列に変換）
    def clean_info(series: pd.Series) -> list[str]:
        return list(map(lambda x: "" if isinstance(x, float) and math.isnan(x) else x, series.tolist()))

    train_info = clean_info(train_df["citation-info"])
    valid_info = clean_info(valid_df["citation-info"])
    test_info = clean_info(test_df["citation-info"])

    # タイトル + sep + 引用文 + sep + 参考文献情報 を連結
    train_X = [t + sep + s + sep + i for t, s, i in zip(train_title, train_X, train_info)]
    valid_X = [t + sep + s + sep + i for t, s, i in zip(valid_title, valid_X, valid_info)]
    test_X = [t + sep + s + sep + i for t, s, i in zip(test_title, test_X, test_info)]

    # ラベルを数値に変換
    def map_labels(series: pd.Series, label_map: dict) -> list[int]:
        return list(map(lambda x: label_map[x], series.tolist()))

    train_role_y = map_labels(train_df["role"], ROLE_MAP)
    train_type_y = map_labels(train_df["type"], TYPE_MAP)
    train_func_y = map_labels(train_df["function"], FUNCTION_MAP)

    valid_role_y = map_labels(valid_df["role"], ROLE_MAP)
    valid_type_y = map_labels(valid_df["type"], TYPE_MAP)
    valid_func_y = map_labels(valid_df["function"], FUNCTION_MAP)

    test_role_y = map_labels(test_df["role"], ROLE_MAP)
    test_type_y = map_labels(test_df["type"], TYPE_MAP)
    test_func_y = map_labels(test_df["function"], FUNCTION_MAP)

    return (
        train_X,
        train_role_y,
        train_type_y,
        train_func_y,
        valid_X,
        valid_role_y,
        valid_type_y,
        valid_func_y,
        test_X,
        test_role_y,
        test_type_y,
        test_func_y,
    )
