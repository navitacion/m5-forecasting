import pickle
import numpy as np

from utils.features import Feature
from utils.utils import load_data

class SellPrice(Feature):
    def create_features(self):
        self.new_colname = ['sell_price']
        self.df[self.new_colname[0]] = self.df[self.new_colname[0]].astype(np.float32)

class Weekday(Feature):
    '''
    曜日を生成する
    '''
    def create_features(self):
        self.new_colname = ['weekday']
        self.df[self.new_colname[0]] = self.df['date'].dt.weekday


class Snap(Feature):
    """
    snapを店がある地域のものに設定する

    """
    def create_features(self):
        self.new_colname = ['snap']
        self.df[self.new_colname[0]] = 0

        self.df.loc[self.df[self.df['state_id'] == 'CA'].index, self.new_colname[0]] = self.df.loc[
            self.df[self.df['state_id'] == 'CA'].index, 'snap_CA']

        self.df.loc[self.df[self.df['state_id'] == 'TX'].index, self.new_colname[0]] = self.df.loc[
            self.df[self.df['state_id'] == 'TX'].index, 'snap_TX']

        self.df.loc[self.df[self.df['state_id'] == 'WI'].index, self.new_colname[0]] = self.df.loc[
            self.df[self.df['state_id'] == 'WI'].index, 'snap_WI']


class Lag(Feature):
    """
    7, 14, 21, 28日前の売上
    """
    def create_features(self):
        self.new_colname = ['lag_7', 'lag_14', 'lag_21', 'lag_28']
        for lag, lagcol in zip([7, 14, 21, 28], self.new_colname):
            self.df[lagcol] = self.df[['id', 'demand']].groupby('id')['demand'].shift(lag)

        for window, lagcol in zip([7, 14, 21, 28], ['lag_7', 'lag_14', 'lag_21', 'lag_28']):
            self.df[lagcol + f'_win_{window}'] = self.df[['id', lagcol]].groupby('id')[lagcol].transform(lambda x: x.rolling(window).mean())
            self.new_colname = self.new_colname + [lagcol + f'_win_{window}']


class RollMean(Feature):
    """
    移動平均7, 14, 21, 28日
    """
    def create_features(self):
        self.new_colname = ['rmean_7', 'rmean_14', 'rmean_21', 'rmean_28']
        for window, wincol in zip([7, 14, 21, 28], self.new_colname):
            self.df[wincol] = self.df[['id', 'demand']].groupby('id')['demand'].transform(lambda x: x.rolling(window).mean())


if __name__ == '__main__':

    save_dir = '../features'

    # Load Data
    # df = load_data(merge=True)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)

    # Weekday(df, dir=save_dir).run().save()
    # Snap(df, dir=save_dir).run().save()
    Lag(df, dir=save_dir).run().save()
    RollMean(df, dir=save_dir).run().save()
