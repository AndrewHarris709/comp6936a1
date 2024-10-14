import re

import pandas as pd


def word_count(df: pd.DataFrame) -> pd.DataFrame:
    df["word_count"] = df['sentence'].str.count(' ')
    df.loc[df["sentence"].str.len() > 0, "word_count"] += 1

    return df


def character_count(df: pd.DataFrame) -> pd.DataFrame:
    df['character_count'] = df['sentence'].str.len()
    return df


def chars_per_word(df: pd.DataFrame) -> pd.DataFrame:
    df['chars_per_word'] = df['sentence'].str.len() / (df['sentence'].str.count(' ') + 1)
    return df


def capital_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['capital_ratio'] = df['sentence'].str.count(r'[A-Z]') / df['sentence'].str.len()
    df.loc[df['sentence'].str.len() == 0, 'capital_ratio'] = 0
    return df


def character_ratio(df: pd.DataFrame, character: str) -> pd.DataFrame:
    df[f'{character}_ratio'] = df['sentence'].str.count(rf'\{character}') / df['sentence'].str.len()
    df.loc[df['sentence'].str.len() == 0, f'{character}_ratio'] = 0
    return df


def character_ratios(df: pd.DataFrame, characters: list[str]) -> pd.DataFrame:
    for character in characters:
        df = character_ratio(df, character)
    return df


def first_year(df: pd.DataFrame) -> pd.DataFrame:
    df['first_year'] = df['sentence'].str.extract(r'(?:[\D]|^)(\d{4})(?:[\D]|$)').fillna(1900).astype("int64")
    return df


def word_instance_ratio(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    df[f'word_{targets[0]}_ratio'] = df['sentence'].str.count(fr"{'|'.join(targets)}", re.I) / (df['sentence'].str.count(' ') + 1)
    return df


def word_instances_ratios(df: pd.DataFrame, targets: list[list[str] | str]) -> pd.DataFrame:
    for target in targets:
        if type(target) is str:
            df = word_instance_ratio(df, [target])
        else:
            df = word_instance_ratio(df, target)
    return df


def word_instance(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    df[f'word_{targets[0]}_count'] = df['sentence'].str.count(fr"{'|'.join(targets)}", re.I)
    return df


def word_instances(df: pd.DataFrame, targets: list[list[str] | str]) -> pd.DataFrame:
    for target in targets:
        if type(target) is str:
            df = word_instance(df, [target])
        else:
            df = word_instance(df, target)
    return df


def word_pair(df: pd.DataFrame, target: list[list[str] | str]) -> pd.DataFrame:
    df = word_instances(df, target)
    first_name = target[0] if type(target[0]) is str else target[0][0]
    second_name = target[1] if type(target[1]) is str else target[1][0]

    df[f'{first_name}{second_name}_word_pair'] = ((df[f'word_{second_name}_count'] -
                                               df[f'word_{first_name}_count'])
                                               / (df['sentence'].str.count(' ') + 1))
    df = df.drop(columns=[f'word_{first_name}_count', f'word_{second_name}_count'])
    return df


def word_pairs(df: pd.DataFrame, target: list[list[list[str] | str]]) -> pd.DataFrame:
    for target in target:
        df = word_pair(df, target)

    return df
