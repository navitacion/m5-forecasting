import glob, pickle, time, datetime
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from utils.preprocessing import preprocessing
from utils.utils import load_data, load_from_feather, reduce_mem_usage
from utils.parameters import *
from model.Model import LGBMModel


# Config  #####################################
config = {
    'features': None,
    'params': lgbm_params,
    'cv': TimeSeriesSplit(n_splits=4),
    'num_boost_round': 200,
    'early_stopping_rounds': 100,
    'verbose': 20,
    'exp_name': 'LightGBM_pre_reg_timeseries'
}

save_model = True
print(config['exp_name'])


def main():
    # Load Data  #####################################
    # From csv
    since = time.time()
    print('Data Loading...')
    # From Original
    # data_dir = '../data/input'
    # df = load_data(nrows=None, merge=True, data_dir=data_dir)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)
    df = reduce_mem_usage(df)
    df = preprocessing(df)
    print(df.columns)
    print(df.head())

    # From Feather
    # target_features = [
    #     'Weekday', 'Snap', 'Lag', 'SellPrice', 'Lag_RollMean',
    #     'TimeFeatures', 'Event', 'Ids', 'Lag_SellPrice'
    # ]
    # target_path = [f'../features/{name}.ftr' for name in target_features]
    # df = load_from_feather(target_path)
    # df.sort_values(by='date', ascending=True, inplace=True)

    # Model Training  #####################################
    lgbm = LGBMModel(df, **config)
    model, importance_df = lgbm.train()

    if save_model:
        with open(f"../models/{config['exp_name']}.pkl", 'wb') as f:
            pickle.dump(model, f)

    # Evaluate  #####################################
    res = lgbm.evaluate()
    sub_name = f"{config['exp_name']}_rmse_{lgbm.score:.3f}.csv"
    res.to_csv(f'../data/output/{sub_name}', index=False)

    # Feature Importance  #####################################
    lgbm.visualize_feature_importance()

    # Time Counting
    erapsedtime = time.time() - since
    s = datetime.timedelta(seconds=erapsedtime)
    print(f'All Times: {str(s)}')


if __name__ == '__main__':
    main()
