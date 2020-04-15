import glob, pickle, time, datetime, argparse
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from utils.preprocessing import preprocessing_1, preprocessing_0
from utils.utils import load_data, load_from_feather, reduce_mem_usage
from utils.parameters import *
from model.Model import LGBMModel


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-obj', '--objective', default='regression', choices=['regression', 'poisson', 'tweedie'])
parser.add_argument('-lr', '--learningrate', type=float, default=0.01)
parser.add_argument('-cv', '--crossval', default='kfold')
parser.add_argument('-nsplit', '--nsplit', type=int, default=4)
parser.add_argument('-num', '--num_boost_round', type=int, default=1000)
parser.add_argument('-early', '--early_stopping_rounds', type=int, default=10)
parser.add_argument('-drate', '--data_rate', type=float, default=0.1)
parser.add_argument('-prep', '--preprocessing', type=int, default=1)
parser.add_argument('--postprocess', action='store_true')
args = parser.parse_args()


# Parameter  #############################################################
params = {
    'boosting_type': 'gbdt',
    'objective': args.objective,
    'metric': 'rmse',
    'learning_rate': args.learningrate
}

# Cross Validation
cv = {
    'kfold': KFold(n_splits=args.nsplit),
    'time': TimeSeriesSplit(n_splits=args.nsplit)
}

prep_dict = {
    0: preprocessing_0,
    1: preprocessing_1,
}


# Config  #####################################
config = {
    'features': None,
    'params': params,
    'cv': cv[args.crossval],
    'num_boost_round': args.num_boost_round,
    'early_stopping_rounds': args.early_stopping_rounds,
    'verbose': 100,
    'use_data': args.data_rate,
    'exp_name': args.expname
}

save_model = True


def main():
    # Load Data  #####################################
    # From csv
    since = time.time()
    print('Data Loading...')
    # From Original  #################
    # data_dir = '../data/input'
    # df = load_data(nrows=None, merge=True, data_dir=data_dir)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)
    df = prep_dict[args.preprocessing](df)
    df = reduce_mem_usage(df)

    # From Feather  #################
    # target_features = [
    #     'Snap', 'SellPrice', 'Lag_RollMean_7', 'Lag_RollMean_14', 'Lag_RollMean_21',
    #     'TimeFeatures', 'Lag_SellPrice', 'Ids'
    # ]
    # target_path = [f'../features/{name}.ftr' for name in target_features]
    # df = load_from_feather(target_path)
    # df = reduce_mem_usage(df)

    # Model Training  #####################################
    lgbm = LGBMModel(df, **config)
    model, importance_df = lgbm.train()

    if save_model:
        with open(f"../models/{config['exp_name']}.pkl", 'wb') as f:
            pickle.dump(model, f)

    # Evaluate  #####################################
    res = lgbm.evaluate(postprocess=args.postprocess)
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
