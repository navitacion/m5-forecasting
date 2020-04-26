import glob, pickle, time, datetime, argparse, gc
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from utils.utils import load_data, load_from_feather, reduce_mem_usage, seed_everything
from model.Model import LGBMModel


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-obj', '--objective', default='regression', choices=['regression', 'poisson', 'tweedie'])
parser.add_argument('-lr', '--learningrate', type=float, default=0.01)
parser.add_argument('-subs', '--subsample', type=float, default=1.0)
parser.add_argument('-featfrac', '--featurefraction', type=float, default=1.0)
parser.add_argument('-cv', '--crossval', default='kfold', choices=['kfold', 'time', 'none'])
parser.add_argument('-nsplit', '--nsplit', type=int, default=4)
parser.add_argument('-num', '--num_boost_round', type=int, default=1000)
parser.add_argument('-early', '--early_stopping_rounds', type=int, default=10)
parser.add_argument('-drate', '--data_rate', type=float, default=0.1)
parser.add_argument('-prep', '--preprocess', action='store_true')
parser.add_argument('-post', '--postprocess', action='store_true')
args = parser.parse_args()


# Parameter  #############################################################
# params = {
#     'boosting_type': 'gbdt',
#     'objective': args.objective,
#     'metric': 'rmse',
#     'learning_rate': args.learningrate,
#     'subsample': args.subsample,
#     'subsample_freq': 1,
#     'feature_fraction': args.featurefraction,
#     'seed': 0
# }

params = {
    'boosting_type': 'gbdt',
    'objective': args.objective,
    'metric': 'rmse',
    'learning_rate': args.learningrate,
    'subsample': args.subsample,
    'subsample_freq': 1,
    'feature_fraction': args.featurefraction,
    'seed': 0,
    'num_leaves': 2**11-1,
    'min_data_in_leaf': 2**12-1,
    'n_estimators': 1400,
    'boost_from_average': False,
}

if args.objective == 'tweedie':
    params.update({'tweedie_variance_power': 1.1})

# Cross Validation
cv = {
    'kfold': KFold(n_splits=args.nsplit),
    'time': TimeSeriesSplit(n_splits=args.nsplit),
    'none': 'none'
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
    'exp_name': args.expname,
    'drop_f': ['snap_CA', 'snap_WI', 'snap_TX', 'cat_id', 'state_id', 'dept_id',
               'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'],
    'use_prep': args.preprocess,
}

save_model = True


def main():
    seed_everything(0)
    # Load Data  #####################################
    # From csv
    since = time.time()
    print('Data Loading...')
    # From Original  #################
    # data_dir = '../data/input'
    # df = load_data(nrows=None, merge=True, data_dir=data_dir)

    # From Pickle  ###################
    # with open('../data/input/data.pkl', 'rb') as f:
    #     df = pickle.load(f)

    # Preprocessing
    # df = prep_dict[args.preprocessing](df)
    # df = reduce_mem_usage(df)

    # From Feather  #################
    target_features = [
        'Snap', 'SellPrice', 'Lag', 'Lag_RollMean_28', 'Lag_RollMean_45',
        'TimeFeatures', 'Lag_SellPrice', 'Lag_SellPrice_diff', 'Ids', 'Event'
    ]

    target_path = [f'../features/{name}.ftr' for name in target_features]
    df = load_from_feather(target_path)

    # Model Training  #####################################
    lgbm = LGBMModel(df, **config)
    model, importance_df = lgbm.train()

    if save_model:
        with open(f"../models/{config['exp_name']}.pkl", 'wb') as f:
            pickle.dump(model, f)

    # WRMSSE  ##################################################
    print('Reading files...')
    calendar = pd.read_csv('../data/input/calendar.csv')
    sell_prices = pd.read_csv('../data/input/sell_prices.csv')
    sales_train_validation = pd.read_csv('../data/input/sales_train_validation.csv')
    train_fold_df = sales_train_validation.iloc[:, :-28]
    valid_fold_df = sales_train_validation.iloc[:, -28:]
    del sales_train_validation

    wrmsse = lgbm.get_wrmsse(train_fold_df, valid_fold_df, calendar, sell_prices)
    print(f'WRMSSE: {wrmsse:.3f}')
    del calendar, sell_prices, train_fold_df, valid_fold_df
    gc.collect()

    # Evaluate  #####################################
    res = lgbm.evaluate(postprocess=args.postprocess)
    sub_name = f"{config['exp_name']}_wrmsse_{wrmsse:.3f}.csv"
    res.to_csv(f'../data/output/{sub_name}', index=False)
    del df
    gc.collect()

    # Feature Importance  #####################################
    lgbm.visualize_feature_importance()

    # Time Counting  ##################################################
    erapsedtime = time.time() - since
    s = datetime.timedelta(seconds=erapsedtime)
    print(f'All Times: {str(s)}')


if __name__ == '__main__':
    main()
