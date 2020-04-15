

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
}

# https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50
lgbm_params_2 = {
    "objective" : "poisson",
    "learning_rate" : 0.01,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": ["rmse"],
    'num_leaves': 50,
    "max_depth": 7,
    "min_data_in_leaf": 50,
}

lgbm_params_3 = {
    "objective" : "tweedie",
    "learning_rate" : 0.01,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": ["rmse"],
    'num_leaves': 50,
    "max_depth": 7,
    "min_data_in_leaf": 50,
}