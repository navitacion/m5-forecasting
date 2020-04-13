import gc, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocessing(df):
    # Date  ##########################################
    df['date'] = pd.to_datetime(df['date'].values)
    df['weekday'] = df['date'].dt.weekday

    # integrate 'snap' feature  ######################
    # def snap(row):
    #     if 'CA' in row['store_id']:
    #         return row['snap_CA']
    #     elif 'TX' in row['store_id']:
    #         return row['snap_TX']
    #     elif 'WI' in row['store_id']:
    #         return row['snap_WI']
    #     else:
    #         pass
    #
    # df['snap'] = df.apply(snap, axis=1)

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

    # Dtypes  ##########################################
    cat_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'snap_CA', 'snap_TX', 'snap_WI', 'weekday', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in cat_cols:
        try:
            df[c] = df[c].astype('category')
        except:
            pass

    return df
