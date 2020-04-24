import gc, os, glob, random
import numpy as np
import pandas as pd


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    with np.errstate(invalid='ignore'):
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def load_data(nrows=None, merge=False, data_dir='../data/input'):
    print('Reading files...')
    calendar = pd.read_csv(os.path.join(data_dir, 'calendar.csv'))
    sell_prices = pd.read_csv(os.path.join(data_dir, 'sell_prices.csv'))
    sales_train_validation = pd.read_csv(os.path.join(data_dir, 'sales_train_validation.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    # melt sales data, get it ready for training
    _ids = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_train_validation = pd.melt(sales_train_validation, id_vars=_ids, var_name='day', value_name='demand')
    sales_train_validation = reduce_mem_usage(sales_train_validation)

    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]

    # change column names
    d_col_1 = [f'd_{x}' for x in np.arange(1914, 1941 + 1, 1)]
    d_col_2 = [f'd_{x}' for x in np.arange(1942, 1969 + 1, 1)]
    d_col_1 = ['id'] + d_col_1
    d_col_2 = ['id'] + d_col_2

    test1.columns = d_col_1
    test2.columns = d_col_2
    del d_col_1, d_col_2

    # get product table
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    # merge with product table
    test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
    test1 = test1.merge(product, how='left', on='id')
    test2 = test2.merge(product, how='left', on='id')
    test2['id'] = test2['id'].str.replace('_validation', '_evaluation')

    test1 = pd.melt(test1, id_vars=_ids, var_name='day', value_name='demand')
    test2 = pd.melt(test2, id_vars=_ids, var_name='day', value_name='demand')

    sales_train_validation['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'

    data = pd.concat([sales_train_validation, test1, test2], axis=0)

    del sales_train_validation, test1, test2, test1_rows, test2_rows, submission
    gc.collect()

    # get only a sample for fst training
    if nrows is not None:
        data = data.loc[nrows:]

    # drop some calendar features
    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)

    # delete test2 for now
    #     data = data[data['part'] != 'test2']

    if merge:
        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data = pd.merge(data, calendar, how='left', left_on=['day'], right_on=['d'])
        data.drop(['d', 'day'], inplace=True, axis=1)
        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    else:
        pass

    data['date'] = pd.to_datetime(data['date'].values)

    data = reduce_mem_usage(data)
    del sell_prices, calendar
    gc.collect()

    return data


def load_from_feather(target_path):

    """
    Loading from Feather
    :param target_features: list
        target features(ex. [Weekday])
    :return: dataframe
        loaded RawData
    """
    df = None

    for i, path in enumerate(target_path):
        d = pd.read_feather(path)
        d = reduce_mem_usage(d, verbose=False)
        if i == 0:
            df = d
        else:
            df = pd.merge(df, d, on=['id', 'date', 'part', 'demand'], how='outer')
            del d
            gc.collect()

    return df
