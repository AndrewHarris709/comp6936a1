# Comp 6936 - Assignment 1
_Andrew Harris - 201932175_


A project exploring the feature engineering potential of GAMs. Three iterations of feature sets are made, with each one improving
on the last. Improvements are made by looking at the learned splines for each feature.

## Running The Code

The code and report for this project is primarily contained in the `analysis2.ipynb` notebook. This notebook contains all three feature iterations including performance metrics and GAM spline plots.

External libraries required for this notebook are included in `requirements.txt`. All requirements can be installed by running: `pip install -r requirements.txt`

**Python 1.12** was used in the development of this code.

## Feature Generation

Specific functions for feature generation can be found in `feature_gen.py`. Each routine takes a pandas DataFrame and returns a DataFrame, so these functions can be used in `.pipe` chains like so: `df.pipe(word_count).pipe(word_instances_ratios, ['low'])`

Basic unit tests for these functions can be found in `tests/test_feature_gen.py`. These tests use the `pytest` framework, which is included in `requirements.txt`.

## .idea Folder

The /.idea folder contains PyCharm project information, in case you want to do some development on this yourself! PyCharm is not required for development on this project.
