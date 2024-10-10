import pandas as pd
from feature_gen import (word_count, character_count, chars_per_word, capital_ratio,
                         dollar_sign_ratio, first_year, word_instance_ratio)

# First Year
# Word Count
# Character Count
# Characters Per Word
# Capital vs. Lowercase letters
# % of '$' signs
# "sold, sell, selling"
# "buy, bought, buying"
# "low, lows, lower, lowers, lowering"
# "high, highs, higher, highest"
# "new, newer, newest"
# "old, older, oldest"
# "grow, growth, grown, growing"
# "shrink, shrunk, shrunken, shrinking"
# "bull, bullish"
# "bear, bearish"

df = pd.read_csv("./data/data.csv").rename(columns={'Sentence': 'sentence', 'Sentiment': 'sentiment'})

df = (df.pipe(first_year)
        .pipe(word_count)
        .pipe(character_count)
        .pipe(chars_per_word)
        .pipe(capital_ratio)
        .pipe(dollar_sign_ratio)
        .pipe(word_instance_ratio, ['sold', 'sell'])
        .pipe(word_instance_ratio, ['buy', 'bought'])
        .pipe(word_instance_ratio, ['low'])
        .pipe(word_instance_ratio, ['high'])
        .pipe(word_instance_ratio, ['new'])
        .pipe(word_instance_ratio, ['old'])
        .pipe(word_instance_ratio, ['grow'])
        .pipe(word_instance_ratio, ['shrink'])
        .pipe(word_instance_ratio, ['bull'])
        .pipe(word_instance_ratio, ['bear']))

df.to_csv("test.csv")
