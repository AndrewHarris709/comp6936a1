import pandas as pd
from matplotlib import pyplot as plt

from feature_gen import (word_count, character_count, chars_per_word, capital_ratio,
                         dollar_sign_ratio, first_year, word_instance_ratio)
from pygam import LinearGAM
import numpy as np

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
      .pipe(word_instance_ratio, ['fall'])
      .pipe(word_instance_ratio, ['bull'])
      .pipe(word_instance_ratio, ['bear'])
      .drop(columns=['sentence']))

df_X = df.drop(['sentiment'], axis=1).values
df_y = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

lams = np.random.rand(5842, 16)
lams = lams * 11 - 3
np.exp(lams)
gam = LinearGAM(n_splines=10).gridsearch(df_X, df_y, lam=lams)

titles = df.drop(['sentiment'], axis=1).columns
fig, ax = plt.subplots(4, 4, figsize=(15, 15))

for i, x in enumerate(ax.flatten()):
    XX = gam.generate_X_grid(term=i)
    x.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    x.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    x.set_title(titles[i])

fig.savefig("./gam.pdf")
