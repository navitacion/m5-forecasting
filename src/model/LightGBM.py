import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


class LGBMModel:

    def __init__(self, train, validation, evaluation, exp_name='LGBM_01'):
        self.train_id = train['id'].values
        self.target = train['values'].values
        self.val_id = validation['id'].values
        self.eval_id = evaluation['id'].values
        self.val_d = validation['d'].values
        self.eval_d = evaluation['d'].values

        X = train.drop(['id', 'd', 'values'], axis=1)
        self.features = X.columns
        self.X = X.values
        self.vals = validation[self.features].values
        self.evals = evaluation[self.features].values

        self.importances = np.zeros((len(self.features)))
        self.importance_df = None
        self.best_score = 10000
        self.models = []
        self.exp_name = exp_name

    def train(self, params, cv, num_boost_round=1000, early_stopping_rounds=20, verbose=200, savemodel=True):
        print('LightGBM Model Training...')
        res_rmse = 0
        for i, (trn_idx, val_idx) in enumerate(cv.split(self.X)):

            train_data = lgb.Dataset(self.X[trn_idx], label=self.target[trn_idx])
            valid_data = lgb.Dataset(self.X[val_idx], label=self.target[val_idx], reference=train_data)

            model = lgb.train(params,
                              train_data,
                              valid_sets=[train_data, valid_data],
                              valid_names=['train', 'eval'],
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=verbose
                              )
            self.models.append(model)

            self.importances += model.feature_importance() / cv.get_n_splits()

            pred = model.predict(self.X[val_idx], num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_true=self.target[val_idx], y_pred=pred))
            res_rmse += rmse / cv.get_n_splits()
            print(f'{i} Fold  RMSE: {rmse:.3f}')
            print('# ' *30)

        self.importance_df = pd.DataFrame({
            'features': self.features,
            'importance': self.importances
        })

        if savemodel:
            model_name = self.exp_name + f'_{res_rmse:.4f}'

            with open(f'../models/{model_name}.pkl', 'wb') as f:
                pickle.dump(self.models, f)

        return self.models, self.importance_df


    def evaluate(self):
        assert len(self.models) != 0, 'Model is not trained...'
        print('Evaluate...')

        pred_val = np.zeros(len(self.val_id))
        pred_eval = np.zeros(len(self.eval_id))

        for model in self.models:
            pred_val += model.predict(self.vals, num_iteration=model.best_iteration) / len(self.models)
            pred_eval += model.predict(self.evals, num_iteration=model.best_iteration) / len(self.models)

        res_val = pd.DataFrame({
            'id': self.val_id,
            'date': self.val_d,
            'pred': pred_val
        })

        res_eval = pd.DataFrame({
            'id': self.eval_id,
            'date': self.eval_d,
            'pred': pred_eval
        })

        # For submit
        print('Creating Submission')
        res = pd.DataFrame()
        F_list = [f'F{i}' for i in range(1, 29, 1)]
        # Validation
        id_list = res_val['id'].unique()
        for _id in tqdm(id_list):
            temp = res_val[res_val['id'] == _id].sort_values(by='date').reset_index(drop=True)
            temp = temp[['pred']].T.reset_index(drop=True)
            temp.columns = F_list
            temp['id'] = _id
            c = ['id'] + F_list
            temp = temp[c]
            res = pd.concat([res, temp], axis=0, ignore_index=True)

        # Evaluation
        id_list = res_eval['id'].unique()
        for _id in tqdm(id_list):
            temp = res_eval[res_eval['id'] == _id].sort_values(by='date').reset_index(drop=True)
            temp = temp[['pred']].T.reset_index(drop=True)
            temp.columns = F_list
            temp['id'] = _id
            c = ['id'] + F_list
            temp = temp[c]
            res = pd.concat([res, temp], axis=0, ignore_index=True)

        print('FINISH')

        return res

    def visualize_feature_importance(self):
        _importance_df = self.importance_df.sort_values(by='importance', ascending=False)
        fig = plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='features', data=_importance_df)
        plt.title('Feature Imporrance')
        plt.show()