from sklearn.model_selection import KFold

from utils.preprocessing import preprocessing
from utils.load_data import load_data
from model.LightGBM import LGBMModel


def main():
    # Load Data  #####################################
    data_dir = '../data'
    train, vals, evals = load_data(data_dir)

    train, vals, evals = preprocessing(train, vals, evals)

    # Config  #####################################
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
    }

    cv = KFold(n_splits=3)
    num_boost_round = 5000
    early_stopping_rounds = 200
    verbose = 1000
    exp_name = 'LGBM_01'

    # Model Setting  #####################################
    print('LightGBM Model Building...')
    lgbm = LGBMModel(train, vals, evals, exp_name)
    model, importance_df = lgbm.train(params, cv, num_boost_round, early_stopping_rounds, verbose, savemodel=True)


    # Evaluate  #####################################
    print('Evaluate...')
    res = lgbm.evaluate()
    res.to_csv('../data/output/submission.csv', index=False)


if __name__ == '__main__':
    main()
