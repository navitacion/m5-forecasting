import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.features import Feature
from utils.utils import load_data


class SellPrice(Feature):
    def create_features(self):
        self.new_colname = ['sell_price']
        self.df[self.new_colname[0]] = self.df[self.new_colname[0]].astype(np.float32)


class TimeFeatures(Feature):
    def create_features(self):
        self.new_colname = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear']
        for c in self.new_colname:
            self.df[c] = getattr(self.df['date'].dt, c).astype(np.int32)

        self.df['year'] = self.df['year'] - self.df['year'].min()
        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.int8)


class Snap(Feature):
    """
    snapを店がある地域のものに設定する
    state_idがCAであれば、snapはsnap_CAの値を使う
    """
    def create_features(self):
        self.new_colname = ['snap', 'snap_sum']
        self.df['snap'] = 0

        self.df.loc[self.df[self.df['state_id'] == 'CA'].index, 'snap'] = self.df.loc[
            self.df[self.df['state_id'] == 'CA'].index, 'snap_CA']

        self.df.loc[self.df[self.df['state_id'] == 'TX'].index, 'snap'] = self.df.loc[
            self.df[self.df['state_id'] == 'TX'].index, 'snap_TX']

        self.df.loc[self.df[self.df['state_id'] == 'WI'].index, 'snap'] = self.df.loc[
            self.df[self.df['state_id'] == 'WI'].index, 'snap_WI']

        self.df['snap_sum'] = self.df['snap_CA'] + self.df['snap_TX'] + self.df['snap_WI']



class Lag(Feature):
    """
    28, 30, 90日前の売上
    lagは28以上に設定すること
    """
    def create_features(self):
        self.new_colname = ['lag_28', 'lag_30', 'lag_90']
        for lag, lagcol in zip([28, 30, 90], self.new_colname):
            self.df[lagcol] = self.df.groupby('id')['demand'].shift(lag)


class Lag_RollMean_28(Feature):
    """
    lagと移動平均を組み合わせ
    lagは28以上に設定すること
    """
    def create_features(self):
        self.new_colname = []
        windows = [28]
        periods = [7, 14, 21, 30, 90]
        for window in windows:
            for period in periods:
                col = f'rolling_{window}_mean_t{period}'
                self.df[col] = self.df.groupby('id')['demand'].transform(
                    lambda x: x.shift(window).rolling(period).mean()).astype(np.float32)
                self.new_colname.append(col)
                col = f'rolling_{window}_std_t{period}'
                self.df[col] = self.df.groupby('id')['demand'].transform(
                    lambda x: x.shift(window).rolling(period).std()).astype(np.float32)
                self.new_colname.append(col)


class Lag_RollMean_45(Feature):
    """
    lagと移動平均を組み合わせ
    lagは28以上に設定すること
    """
    def create_features(self):
        self.new_colname = []
        windows = [45]
        periods = [7, 14, 21, 30, 90]
        for window in windows:
            for period in periods:
                col = f'rolling_{window}_mean_t{period}'
                self.df[col] = self.df.groupby('id')['demand'].transform(
                    lambda x: x.shift(window).rolling(period).mean()).astype(np.float32)
                self.new_colname.append(col)
                col = f'rolling_{window}_std_t{period}'
                self.df[col] = self.df.groupby('id')['demand'].transform(
                    lambda x: x.shift(window).rolling(period).std()).astype(np.float32)
                self.new_colname.append(col)

class Event(Feature):
    """
    イベント情報
    """
    def create_features(self):
        self.new_colname = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for f in self.new_colname:
            self.df[f] = self.df[f].astype('object')
            self.df[f].fillna('unknown', inplace=True)
            # イベントが発生したかどうか
            if f == 'event_name_1':
                self.df['isEvent'] = self.df['event_name_1'].apply(lambda x: 0 if x == 'unknown' else 1)
                self.df['isEvent'] = self.df['isEvent'].astype('category')
                self.new_colname.append('isEvent')

            encoder = LabelEncoder()
            self.df[f] = encoder.fit_transform(self.df[f])

        # 前日にイベントがあったかどうか
        self.df['isEvent_past1'] = self.df.groupby('id')['isEvent'].transform(lambda x: x.shift(1))
        self.new_colname.append('isEvent_past1')


class Ids(Feature):
    """
    各種ID
    """
    def create_features(self):
        self.new_colname = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        for f in self.new_colname:
            encoder = LabelEncoder()
            self.df[f] = encoder.fit_transform(self.df[f])


class Lag_SellPrice(Feature):
    """
    過去の価格
    lagで何日前の値を引っ張ってくるか指定
    """
    def create_features(self):
        self.new_colname =[]
        self.df['sell_price'] = self.df['sell_price'].astype(np.float32)
        lags = [7, 30, 90, 365]
        for lag in lags:
            col = f'sell_price_lag_{lag}'
            self.df[col] = self.df[['id', 'sell_price']].groupby('id')['sell_price'].transform(lambda x: x.shift(lag))
            self.new_colname.append(col)


class Lag_SellPrice_diff(Feature):
    """
    過去の価格からの差分と割合
    """
    def create_features(self):
        self.new_colname =[]
        self.df['sell_price'] = self.df['sell_price'].astype(np.float32)
        lags = [1, 7, 30, 90, 365]
        for lag in lags:
            col = f'sell_price_lag_{lag}_diff'
            self.df[col] = self.df[['id', 'sell_price']].groupby('id')['sell_price'].transform(lambda x: x - x.shift(lag))
            self.new_colname.append(col)
            col = f'sell_price_lag_{lag}_div'
            self.df[col] = self.df[['id', 'sell_price']].groupby('id')['sell_price'].transform(lambda x: x / x.shift(lag))
            self.new_colname.append(col)


# class Price_fe(Feature):
#     """
#     店舗・商品ごとの価格の基礎統計量
#     """
#     def create_features(self):
#         self.new_colname = ['price_max', 'price_min', 'price_mean', 'price_std', 'price_norm']
#         self.df['price_max'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
#         self.df['price_min'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
#         self.df['price_mean'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
#         self.df['price_std'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
#         self.df['price_norm'] = self.df['sell_price'] / self.df['price_max']
#         for c in self.new_colname:
#             self.df[c] = self.df[c].astype(np.float32)


class Price_WeekNum(Feature):
    """
    週ごとの店舗・商品価格の基礎統計量
    """
    def create_features(self):
        self.new_colname = ['price_week_max', 'price_week_min', 'price_week_mean', 'price_week_std']
        self.df['price_week_max'] = self.df.groupby(['store_id', 'item_id', 'wm_yr_wk'])['sell_price'].transform('max')
        self.df['price_week_min'] = self.df.groupby(['store_id', 'item_id', 'wm_yr_wk'])['sell_price'].transform('min')
        self.df['price_week_mean'] = self.df.groupby(['store_id', 'item_id', 'wm_yr_wk'])['sell_price'].transform('mean')
        self.df['price_week_std'] = self.df.groupby(['store_id', 'item_id', 'wm_yr_wk'])['sell_price'].transform('std')
        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.float32)


if __name__ == '__main__':

    save_dir = '../features'

    # Load Data
    # df = load_data(merge=True)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)
    print(df.dtypes)

    # SellPrice(df, dir=save_dir).run().save()
    # TimeFeatures(df, dir=save_dir).run().save()
    # Snap(df, dir=save_dir).run().save()
    # Lag(df, dir=save_dir).run().save()
    # Lag_RollMean_28(df, dir=save_dir).run().save()
    Lag_RollMean_45(df, dir=save_dir).run().save()
    # Event(df, dir=save_dir).run().save()
    # Ids(df, dir=save_dir).run().save()
    # Lag_SellPrice(df, dir=save_dir).run().save()
    # Lag_SellPrice_diff(df, dir=save_dir).run().save()
    # Price_fe(df, dir=save_dir).run().save()
    # Price_WeekNum(df, dir=save_dir).run().save()
