import gc, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from .costFunc import custom_asymmetric_train, custom_asymmetric_valid
from .WRMSSE import WRMSSEEvaluator


class M5Model(metaclass=ABCMeta):
    def __init__(self, df, features, params, cv, num_boost_round=1000,
                 early_stopping_rounds=20, verbose=200, exp_name='Model', use_data=None, drop_f=None):

        self.params = params
        self.cv = cv
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping_rounds
        self.verbose = verbose
        self.exp_name = exp_name

        # featuresがNoneの場合はすべての変数を使う
        if features is None:
            features = [c for c in df.columns if c not in ['id', 'part', 'date', 'demand', 'wm_yr_wk']]
        # drop_fで使用しない変数を指定する
        if drop_f is not None:
            features = [c for c in features if c not in drop_f]

        self.features = features
        self.cat_features = [c for c in df.select_dtypes('category').columns if c not in self.features]
        self.cat_features = [c for c in self.cat_features if c not in drop_f]

        # Train Data
        self.X = df[df['part'] == 'train'].copy()
        # 価格がないものは販売していないため除外する
        self.X.dropna(subset=['sell_price'], inplace=True)
        # 日付昇順に並び替える
        self.X.sort_values(by='date', ascending=True, inplace=True)
        self.X.reset_index(drop=True, inplace=True)
        # 使用するデータ量を調整
        _limit = int(len(self.X) * (1 - use_data))
        self.X = self.X.iloc[_limit:].reset_index(drop=True)

        self.train_id = self.X['id'].values
        self.target = self.X['demand'].values
        self.X = self.X[self.features].values

        # Validation
        self.vals = df[df['part'] == 'test1'].copy()
        self.val_id = self.vals['id'].values
        self.val_date = self.vals['date'].values
        self.vals = self.vals[self.features].values

        # Evaluation
        self.evals = df[df['part'] == 'test2'].copy()
        self.eval_id = self.evals['id'].values
        self.eval_date = self.evals['date'].values
        self.evals = self.evals[self.features].values

        # WRMSSEのためのvalデータ
        self.X_val_wrmsse = df[(df['date'] > '2016-03-27') & (df['date'] <= '2016-04-24')].copy()
        self.y_val_wrmsse = self.X_val_wrmsse['demand'].values

        self.importances = np.zeros((len(self.features)))
        self.importance_df = None
        self.best_score = 10000
        self.models = []

        del df
        gc.collect()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    def visualize_feature_importance(self, savefig=True):
        assert len(self.models) != 0, 'Model is not trained...'
        _importance_df = self.importance_df.sort_values(by='importance', ascending=False)
        fig = plt.figure(figsize=(12, int(0.8 * _importance_df.shape[0])), facecolor='w')
        sns.barplot(x='importance', y='features', data=_importance_df)
        plt.title('Feature Importance')
        if savefig:
            plt.savefig(f'../fig/{self.exp_name}.png')

    def get_wrmsse(self, train_fold_df, valid_fold_df, calendar, sell_prices):
        """
        WRMSSEの計算
        Parameters
        ----------
        train_fold_df: 元の学習データ
        valid_fold_df
        calendar
        sell_prices

        Returns WRMMSE
        -------

        """
        self.X_val_wrmsse = pd.pivot(self.X_val_wrmsse, index='id', columns='date', values='demand').reset_index()
        self.X_val_wrmsse.columns = ['id'] + ['d_' + str(i) for i in range(1886, 1914)]
        x_val = train_fold_df[['id']].merge(self.X_val_wrmsse, on='id')
        x_val.drop('id', axis=1, inplace=True)
        evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
        score = evaluator.score(x_val)
        return score


class LGBMModel(M5Model):

    def train(self):
        print('LightGBM Model Training...')
        print('Train Data Shape: ', self.X.shape)
        self.score = 0.0

        # クロスバリデーションを用いる場合
        if self.cv != 'none':
            val_pred = np.zeros(len(self.X_val_wrmsse))
            for i, (trn_idx, val_idx) in enumerate(self.cv.split(self.X)):
                train_data = lgb.Dataset(self.X[trn_idx], label=self.target[trn_idx],
                                         categorical_feature=self.cat_features)
                valid_data = lgb.Dataset(self.X[val_idx], label=self.target[val_idx],
                                         categorical_feature=self.cat_features, reference=train_data)

                model = lgb.train(self.params,
                                  train_data,
                                  valid_sets=[valid_data, train_data],
                                  valid_names=['eval', 'train'],
                                  num_boost_round=self.num_boost_round,
                                  early_stopping_rounds=self.early_stopping,
                                  verbose_eval=self.verbose,
                                  fobj=custom_asymmetric_train,
                                  feval=custom_asymmetric_valid
                                  )
                self.models.append(model)

                self.importances += model.feature_importance() / self.cv.get_n_splits()

                pred = model.predict(self.X[val_idx], num_iteration=model.best_iteration)
                rmse = np.sqrt(mean_squared_error(y_true=self.target[val_idx], y_pred=pred))
                self.score += rmse / self.cv.get_n_splits()
                print(f'{i + 1} Fold  RMSE: {rmse:.3f}')
                print('#' * 30)

                val_pred += model.predict(self.X_val_wrmsse[self.features], num_iteration=model.best_iteration) / self.cv.get_n_splits()
                del model, train_data, valid_data, pred, rmse
                gc.collect()
        # クロスバリデーションをしない場合
        else:
            dev_idx = int(len(self.X) * 0.8)
            train_data = lgb.Dataset(self.X[:dev_idx], label=self.target[:dev_idx])
            valid_data = lgb.Dataset(self.X[dev_idx:], label=self.target[dev_idx:],
                                     reference=train_data)

            model = lgb.train(self.params,
                              train_data,
                              valid_sets=[valid_data, train_data],
                              valid_names=['eval', 'train'],
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping,
                              verbose_eval=self.verbose,
                              fobj=custom_asymmetric_train,
                              feval=custom_asymmetric_valid
                              )
            self.models.append(model)

            self.importances += model.feature_importance()

            pred = model.predict(self.X[dev_idx:], num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_true=self.target[dev_idx:], y_pred=pred))
            self.score += rmse
            print(f'RMSE: {rmse:.3f}')
            print('#' * 30)

            val_pred = model.predict(self.X_val_wrmsse[self.features], num_iteration=model.best_iteration)
            del model, train_data, valid_data, pred, rmse
            gc.collect()

        print(f'All Fold RMSE: {self.score:.3f}')

        self.importance_df = pd.DataFrame({
            'features': self.features,
            'importance': self.importances
        })

        self.X_val_wrmsse['demand'] = val_pred
        self.X_val_wrmsse = self.X_val_wrmsse[['id', 'date', 'demand']]

        return self.models, self.importance_df

    def evaluate(self, postprocess=False):
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

        if postprocess:
            alphas = [1.035, 1.03, 1.025]
            weights = [1 / len(alphas)] * len(alphas)
            _res = res.copy()
            for f in F_list:
                _res[f] = 0

                for alpha, weight in zip(alphas, weights):
                    _res[f] += alpha * weight * res[f]

            return _res

        else:
            return res
