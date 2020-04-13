import gc, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .utils import reduce_mem_usage


def preprocessing(df):
    # Date  ##########################################
    new_colname = ['year', 'month', 'quarter', 'week', 'day', 'dayofweek', 'dayofyear', 'weekday']
    for c in new_colname:
        df[c] = getattr(df['date'].dt, c).astype(np.int32)

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
    new_colname = ['lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_30', 'lag_90']
    for lag, lagcol in zip([7, 14, 21, 28, 30, 90], new_colname):
        df[lagcol] = df[['id', 'demand']].groupby('id')['demand'].shift(lag)

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    window = 28
    periods = [7, 14, 21, 30, 90]
    for period in periods:
        df[f'rolling_{window}_mean_t{period}'] = df[['id', 'demand']].groupby('id')['demand'] \
            .transform(lambda x: x.shift(window).rolling(period).mean())
        df[f'rolling_{window}_std_t{period}'] = df[['id', 'demand']].groupby('id')['demand'] \
            .transform(lambda x: x.shift(window).rolling(period).std())

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

    # Lag - Sell_price  ############################################
    df['sell_price'] = df['sell_price'].astype(np.float32)
    lags = [1, 2, 3, 7, 14]
    for lag in lags:
        col = f'sell_price_lag_{lag}'
        df[col] = df[['id', 'sell_price']].groupby('id')['sell_price'].shift(lag)

    df = reduce_mem_usage(df, verbose=False)
    gc.collect()

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
