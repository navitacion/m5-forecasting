import glob, pickle
import pandas as pd
from sklearn.model_selection import KFold

from utils.preprocessing import preprocessing
from utils.utils import load_data, load_from_feather
from utils.parameters import *
from model.Model import LGBMModel



# Config  #####################################
config = {
    'features': None,
    'params': lgbm_params_2,
    'cv': KFold(n_splits=3, shuffle=True),
    'num_boost_round': 100,
    'early_stopping_rounds': 10,
    'verbose': 20,
    'exp_name': 'LightGBM_03'
}

save_model = True


def main():
    # Load Data  #####################################
    print('Data Loading...')
    # From Original
    # data_dir = '../data/input'
    # df = load_data(nrows=None, merge=True, data_dir=data_dir)

    # From Feather
    target_features = ['Weekday', 'Snap', 'Lag', 'SellPrice', 'Lag_RollMean', 'TimeFeatures', 'Event']
    target_path = [f'../features/{name}.ftr' for name in target_features]
    df = load_from_feather(target_path)

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


if __name__ == '__main__':
    main()
