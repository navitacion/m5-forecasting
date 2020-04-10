import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
import lightgbm as lgb



class M5Model(metaclass=ABCMeta):
    def __init__(self, df, features, params, cv, num_boost_round=1000,
                 early_stopping_rounds=20, verbose=200, exp_name='Model'):

        self.params = params
        self.cv = cv
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping_rounds
        self.verbose = verbose
        self.exp_name = exp_name

        train = df[df['part'] == 'train']
        self.train_id = train['id'].values
        self.target = train['demand'].values

        validation = df[df['part'] == 'test1']
        self.val_id = validation['id'].values
        self.val_date = validation['date'].values

        evaluation = df[df['part'] == 'test2']
        self.eval_id = evaluation['id'].values
        self.eval_date = evaluation['date'].values

        self.features = features
        self.X = train[self.features].values
        self.vals = validation[self.features].values
        self.evals = evaluation[self.features].values

        del train, validation, evaluation
        gc.collect()

        self.importances = np.zeros((len(self.features)))
        self.importance_df = None
        self.best_score = 10000
        self.models = []

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    def visualize_feature_importance(self, savefig=True):
        assert len(self.models) != 0, 'Model is not trained...'
        _importance_df = self.importance_df.sort_values(by='importance', ascending=False)
        fig = plt.figure(figsize=(12, int(2 * _importance_df.shape[0])), facecolor='w')
        sns.barplot(x='importance', y='features', data=_importance_df)
        plt.title('Feature Imporrance')
        if savefig:
            plt.savefig(f'../fig/{self.exp_name}.png')
        plt.show()


class LGBMModel(M5Model):

    def train(self):
        print('LightGBM Model Training...')
        self.score = 0.0
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(self.X)):
            train_data = lgb.Dataset(self.X[trn_idx], label=self.target[trn_idx])
            valid_data = lgb.Dataset(self.X[val_idx], label=self.target[val_idx], reference=train_data)

            model = lgb.train(self.params,
                              train_data,
                              valid_sets=[train_data, valid_data],
                              valid_names=['train', 'eval'],
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping,
                              verbose_eval=self.verbose
                              )
            self.models.append(model)

            self.importances += model.feature_importance() / self.cv.get_n_splits()

            pred = model.predict(self.X[val_idx], num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_true=self.target[val_idx], y_pred=pred))
            self.score += rmse / self.cv.get_n_splits()
            print(f'{i + 1} Fold  RMSE: {rmse:.3f}')
            print('#' * 30)

        print(f'All Fold RMSE: {self.score:.3f}')

        self.importance_df = pd.DataFrame({
            'features': self.features,
            'importance': self.importances
        })

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
            'date': self.val_date,
            'demand': pred_val
        })

        res_val = pd.pivot(res_val, index='id', columns='date', values='demand').reset_index()

        res_eval = pd.DataFrame({
            'id': self.eval_id,
            'date': self.eval_date,
            'demand': pred_eval
        })

        res_eval = pd.pivot(res_eval, index='id', columns='date', values='demand').reset_index()

        F_list = [f'F{i + 1}' for i in range(28)]

        res_val.columns = ['id'] + F_list
        res_eval.columns = ['id'] + F_list

        res = pd.concat([res_val, res_eval], axis=0)
        print('FINISH')

        return res
