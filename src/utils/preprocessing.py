import gc, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .utils import reduce_mem_usage


def preprocessing_0(df):
    # NaN  ############################################
    cols = {'event_name_1': 'Nodata',
            'event_type_1': 'Nodata',
            'event_name_2': 'Nodata',
            'event_type_2': 'Nodata'}
    df.fillna(cols, inplace=True)

    # LabelEncoder  ####################################
    lbl_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in lbl_cols:
        lbl = LabelEncoder()
        df[c] = lbl.fit_transform(df[c].values)

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    # Dtypes  ##########################################
    cat_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'snap_CA', 'snap_TX', 'snap_WI', 'weekday', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in cat_cols:
        try:
            df[c] = df[c].astype('category')
        except:
            pass

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    return df


def preprocessing_1(df):
    # Date  ##########################################
    new_colname = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear']
    for c in new_colname:
        df[c] = getattr(df['date'].dt, c).astype(np.int16)

    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # integrate 'snap' feature  ######################
    df['snap'] = 0
    df.loc[df[df['state_id'] == 'CA'].index, 'snap'] = df.loc[
        df[df['state_id'] == 'CA'].index, 'snap_CA']

    df.loc[df[df['state_id'] == 'TX'].index, 'snap'] = df.loc[
        df[df['state_id'] == 'TX'].index, 'snap_TX']

    df.loc[df[df['state_id'] == 'WI'].index, 'snap'] = df.loc[
        df[df['state_id'] == 'WI'].index, 'snap_WI']

    df = reduce_mem_usage(df)
    gc.collect()

    # Lag  ############################################
    # 変数は, すべてlagを28以上にして, F1~F28の予測を1つのモデルで表現するのが目的。
    # https://www.kaggle.com/mfjwr1/for-japanese-beginner-with-wrmsse-in-lgbm
    lags = [28, 29]
    periods = [7, 28]
    for lag in lags:
        df[f'rolling_{lag}'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag))
        for period in periods:
            df[f'rolling_{lag}_mean_t{period}'] = df.groupby(['id'])['demand'] \
                .transform(lambda x: x.shift(lag).rolling(period).mean()).astype(np.float32)
            df[f'rolling_{lag}_std_t{period}'] = df[['id', 'demand']].groupby('id')['demand'] \
                .transform(lambda x: x.shift(lag).rolling(period).std()).astype(np.float32)

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    # Events  ############################################
    cols = {'event_name_1': 'Nodata',
            'event_type_1': 'Nodata',
            'event_name_2': 'Nodata',
            'event_type_2': 'Nodata'}
    df.fillna(cols, inplace=True)
    # イベントがあるかどうか
    # そこまで効果なさそう
    # df['isEvent'] = df['event_name_1'].apply(lambda x: 0 if x == 'Nodata' else 1)

    # LabelEncoder  ####################################
    lbl_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in lbl_cols:
        lbl = LabelEncoder()
        df[c] = lbl.fit_transform(df[c].values)

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    # Dtypes  ##########################################
    cat_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'snap_CA', 'snap_TX', 'snap_WI', 'weekday', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'isEvent']
    for c in cat_cols:
        try:
            df[c] = df[c].astype('category')
        except:
            pass

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    return df


# if __name__ == '__main__':
#     with open('../data/input/data.pkl', 'rb') as f:
#         df = pickle.load(f)
#     df = reduce_mem_usage(df)
#     df = preprocessing_1(df)
#
#     with open(f"../data/input/prep_data.pkl", 'wb') as f:
#         pickle.dump(df, f)
