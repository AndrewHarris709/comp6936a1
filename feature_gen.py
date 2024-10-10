import re

import pandas as pd


def word_count(df: pd.DataFrame) -> pd.DataFrame:
    df["word_count"] = df['sentence'].str.count(' ')
    df["word_count"][df["sentence"].str.len() > 0] += 1

    return df


def character_count(df: pd.DataFrame) -> pd.DataFrame:
    df['character_count'] = df['sentence'].str.len()
    return df


def chars_per_word(df: pd.DataFrame) -> pd.DataFrame:
    df['chars_per_word'] = df['sentence'].str.len() / (df['sentence'].str.count(' ') + 1)
    return df


def capital_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['capital_ratio'] = df['sentence'].str.count(r'[A-Z]') / df['sentence'].str.len()
    df['capital_ratio'][df['sentence'].str.len() == 0] = 0
    return df


def dollar_sign_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['dollar_sign_ratio'] = df['sentence'].str.count(r'\$') / df['sentence'].str.len()
    df['dollar_sign_ratio'][df['sentence'].str.len() == 0] = 0
    return df


def first_year(df: pd.DataFrame) -> pd.DataFrame:
    df['first_year'] = df['sentence'].str.extract(r'(?:[\D]|^)(\d{4})(?:[\D]|$)').fillna(1900).astype(int)
    return df


def word_instance_ratio(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    df[f'word_{targets[0]}_ratio'] = df['sentence'].str.count(fr"{'|'.join(targets)}", re.I) / (df['sentence'].str.count(' ') + 1)
    return df
