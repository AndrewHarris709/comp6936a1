import pandas as pd
from pandas.testing import assert_series_equal

from pytest import fixture
from feature_gen import (word_count, character_count, chars_per_word, capital_ratio, character_ratio, first_year,
                         word_instance_ratio, word_pairs, word_instances)

@fixture
def general_data():
    data = pd.DataFrame([
        "Test Sentence",
        "$APPL: This is the Apple Corporation.",
        "$$One-Long-Word",
        "no capitals",
        "",
    ], columns=["sentence"])
    return data


@fixture
def bighuge_data():
    data = pd.DataFrame([
        "This is a big deal.",
        "This is big, and also huge.",
        "This is a small deal.",
        "Bigger than ever, huger than everything, oh my!",
        "",
    ], columns=["sentence"])
    return data


def test_word_count(general_data):
    result = word_count(general_data)
    assert_series_equal(result['word_count'], pd.Series([2, 6, 1, 2, 0], name='word_count'))


def test_character_count(general_data):
    result = character_count(general_data)
    assert_series_equal(result['character_count'], pd.Series([13, 37, 15, 11, 0], name='character_count'))


def test_chars_per_word(general_data):
    result = chars_per_word(general_data)
    assert_series_equal(result['chars_per_word'], pd.Series([13/2, 37/6, 15/1, 11/2, 0], name='chars_per_word'))


def test_capital_ratio(general_data):
    result = capital_ratio(general_data)
    assert_series_equal(result['capital_ratio'], pd.Series([2/13, 7/37, 3/15, 0, 0], name='capital_ratio'))


def test_char_count(general_data):
    result = character_ratio(general_data, '$')
    assert_series_equal(result[f'$_ratio'], pd.Series([0, 1/37, 2/15, 0, 0], name='$_ratio'))


def test_word_pair():
    data = pd.DataFrame([
        "This is very good!",
        "This is bad, really awful, not good. Did I mention bad?",
        "Bad in some ways, good in others.",
        "I'm feeling neutral about this.",
        "",
    ], columns=["sentence"])
    result = word_pairs(data, [[['bad', 'awful'], 'good'], ['good', ['bad', 'awful']]])
    assert_series_equal(result['badgood_word_pair'], pd.Series([1/4, -2/11, 0, 0, 0], name='badgood_word_pair'))
    assert_series_equal(result['goodbad_word_pair'], pd.Series([-1/4, 2/11, 0, 0, 0], name='goodbad_word_pair'))


def test_first_year():
    data = pd.DataFrame([
        "The year is 2008.",
        "1995 is the year, not 2003.",
        "It is not 20004, it is 2005",
        "There is no year.",
        "",
    ], columns=["sentence"])
    result = first_year(data)
    assert_series_equal(result['first_year'], pd.Series([2008, 1995, 2005, 1900, 1900], name='first_year'))


def test_word_instance_ratio(bighuge_data):
    result = word_instance_ratio(bighuge_data, ["big", "huge"])
    assert_series_equal(result['word_big_ratio'], pd.Series([1/5, 2/6, 0, 2/8, 0], name='word_big_ratio'))


def test_word_instances(bighuge_data):
    result = word_instances(bighuge_data, [['big', 'huge']])
    assert_series_equal(result['word_big_count'], pd.Series([1, 2, 0, 2, 0], name='word_big_count'))
