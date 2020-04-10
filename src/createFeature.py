import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

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


class TimeFeatures(Feature):

    def create_features(self):
        self.new_colname = ['year', 'month', 'quarter', 'week', 'day', 'dayofweek', 'dayofyear']
        for c in self.new_colname:
            self.df[c] = getattr(self.df['date'].dt, c).astype(np.int32)


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
    7, 14, 21, 28, 30, 90日前の売上
    ['lag_7', 'lag_14', 'lag_21', 'lag_28']
    """
    def create_features(self):
        self.new_colname = ['lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_30', 'lag_90']
        for lag, lagcol in zip([7, 14, 21, 28, 30, 90], self.new_colname):
            self.df[lagcol] = self.df[['id', 'demand']].groupby('id')['demand'].shift(lag)

class Lag_RollMean(Feature):
    """
    lagと移動平均を組み合わせ
    """
    def create_features(self):
        self.new_colname = []
        window = 28
        periods = [7, 14, 21, 30, 90]
        for period in periods:
            self.df[f'rolling_{window}_mean_t{period}'] = self.df[['id', 'demand']].groupby('id')['demand']\
                .transform(lambda x: x.shift(window).rolling(period).mean())
            self.new_colname.append(f'rolling_{window}_mean_t{period}')
            self.df[f'rolling_{window}_std_t{period}'] = self.df[['id', 'demand']].groupby('id')['demand'] \
                .transform(lambda x: x.shift(window).rolling(period).std())
            self.new_colname.append(f'rolling_{window}_std_t{period}')


class Event(Feature):
    """
    イベント情報
    """
    def create_features(self):
        self.new_colname = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for f in self.new_colname:
            self.df[f].fillna('unknown', inplace=True)
            encoder = LabelEncoder()
            self.df[f] = encoder.fit_transform(self.df[f])


class Ids(Feature):
    """
    各種ID
    """
    def create_features(self):
        self.colname = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        for f in self.new_colname:
            encoder = LabelEncoder()
            self.df[f] = encoder.fit_transform(self.df[f])


if __name__ == '__main__':

    save_dir = '../features'

    # Load Data
    # df = load_data(merge=True)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)

    # SellPrice(df, dir=save_dir).run().save()
    # Weekday(df, dir=save_dir).run().save()
    # TimeFeatures(df, dir=save_dir).run().save()
    # Snap(df, dir=save_dir).run().save()
    # Lag(df, dir=save_dir).run().save()
    Lag_RollMean(df, dir=save_dir).run().save()
    Event(df, dir=save_dir).run().save()
    Ids(df, dir=save_dir).run().save()
