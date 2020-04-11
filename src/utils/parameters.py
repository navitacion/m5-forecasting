

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
}

# https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50
lgbm_params_2 = {
    "objective" : "poisson",
    "force_row_wise" : True,
    "learning_rate" : 0.075,
    "sub_row" : 0.75,
    "bagging_freq" : 1,
    "lambda_l2" : 0.1,
    "metric": ["rmse"],
    'verbosity': 1,
    'num_leaves': 128,
    "min_data_in_leaf": 50,
}