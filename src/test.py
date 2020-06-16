import gc, os, pickle, datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.utils import reduce_mem_usage

df = pd.read_csv('../data/output/lgbm_tweedie_11_wrmsse_0.562.csv')


alphas = [1.035, 1.03, 1.025]
weights = [1 / len(alphas)] * len(alphas)
_df = df.copy()

F_list = [f'F{i + 1}' for i in range(28)]

for f in F_list:
    _df[f] = 0

    for alpha, weight in zip(alphas, weights):
        _df[f] += alpha * weight * df[f]

_df.to_csv('../data/output/lgbm_tweedie_11_wrmsse_0.562_post.csv', index=False)
