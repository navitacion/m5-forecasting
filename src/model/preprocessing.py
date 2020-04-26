import gc, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocessing_0(df):

    # demandが0で連続している場合を削除
    # sell_priceをnanとすることでモデルを作成する際の除外対象とする
    df['flag'] = df.groupby('id')['demand'].transform(lambda x: x.shift(1)) + \
                 df.groupby('id')['demand'].transform(lambda x: x.shift(-1))
    # 前後の値が0の場合は削除する
    df = df[df['flag'] != 0]
    del df['flag']
    return df


