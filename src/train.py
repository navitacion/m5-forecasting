import glob, pickle, time, datetime
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from utils.preprocessing import preprocessing, preprocessing_2
from utils.utils import load_data, load_from_feather, reduce_mem_usage
from utils.parameters import *
from model.Model import LGBMModel


# Config  #####################################
config = {
    'features': None,
    'params': lgbm_params,
    'cv': TimeSeriesSplit(n_splits=4),
    'num_boost_round': 20000,
    'early_stopping_rounds': 300,
    'verbose': 1000,
    'use_data': 0.5,
    'exp_name': 'LightGBM_pre_reg_timeseries_4'
}

print('LR=0.05-0.01 num_boost_round=200-20000 use_data=0.05-0.3')

save_model = True
print(config['exp_name'])


def main():
    # Load Data  #####################################
    # From csv
    since = time.time()
    print('Data Loading...')
    # From Original  #################
    # data_dir = '../data/input'
    # df = load_data(nrows=None, merge=True, data_dir=data_dir)

    # with open('../data/input/data.pkl', 'rb') as f:
    #     df = pickle.load(f)
    # df = preprocessing_2(df)
    # df = reduce_mem_usage(df)

    # From Feather  #################
    target_features = [
        'Snap', 'SellPrice', 'Lag_RollMean_7', 'Lag_RollMean_14', 'Lag_RollMean_21',
        'TimeFeatures', 'Lag_SellPrice', 'Ids'
    ]
    target_path = [f'../features/{name}.ftr' for name in target_features]
    df = load_from_feather(target_path)
    df = reduce_mem_usage(df)

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
