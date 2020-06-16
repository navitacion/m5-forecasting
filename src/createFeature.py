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
    """
    日付の特徴量
    """
    def create_features(self):
        self.new_colname = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear']
        for c in self.new_colname:
            self.df[c] = getattr(self.df['date'].dt, c).astype(np.int32)

        self.df['year'] = self.df['year'] - self.df['year'].min()
        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.int8)

        self.df['weeknum'] = self.df['date'].apply(lambda x: x.isocalendar()[1])
        self.new_colname.append('weeknum')

        self.df['tm_wm'] = self.df['day'].apply(lambda x: np.ceil(x/7)).astype(np.int8)
        self.new_colname.append('tm_wm')

        self.df['weekend'] = (self.df['dayofweek']>=5).astype(np.int8)
        self.new_colname.append('weekend')


class Snap(Feature):
    """
    snapを店がある地域のものに設定する
    state_idがCAであれば、snapはsnap_CAの値を使う
    """
    def create_features(self):
        self.new_colname = ['snap_CA', 'snap_TX', 'snap_WI', 'snap', 'snap_sum']
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
    28, 60, 90, 180, 365日前の売上数
    lagは28以上に設定すること
    """
    def create_features(self):
        lags = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 60, 90, 180, 365]
        self.new_colname = []
        for lag in lags:
            self.df[f'lag_{lag}'] = self.df.groupby('id')['demand'].shift(lag).astype(np.float32)
            self.new_colname.append(f'lag_{lag}')

            # 差分や割合を計算するとうまく学習しない
            # col = f'demand_lag_{lag}_diff'
            # self.df[col] = self.df['demand'] - self.df[f'lag_{lag}']
            # self.new_colname.append(col)
            # col = f'demand_lag_{lag}_div'
            # self.df[col] = self.df['demand'] / self.df[f'lag_{lag}']
            # self.new_colname.append(col)


class Lag_RollMean_28(Feature):
    """
    lagと移動平均を組み合わせ
    lagは28以上に設定すること
    """
    def create_features(self):
        self.new_colname = []
        windows = [28]
        periods = [7, 14, 30, 60, 90, 180, 365]
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
        periods = [7, 14, 30, 60, 365]
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
        lags = [28, 90, 180, 365]
        for lag in lags:
            col = f'sell_price_lag_{lag}'
            self.df[col] = self.df.groupby('id')['sell_price'].transform(lambda x: x.shift(lag))
            self.new_colname.append(col)


class Lag_SellPrice_diff(Feature):
    """
    過去の価格からの差分と割合
    """
    def create_features(self):
        self.new_colname =[]
        self.df['sell_price'] = self.df['sell_price'].astype(np.float32)
        lags = [28, 90, 180, 365]
        for lag in lags:
            self.df[f'sell_price_lag_{lag}'] = self.df.groupby('id')['sell_price'].transform(lambda x: x.shift(lag)).astype(np.float32)
            col = f'sell_price_lag_{lag}_diff'
            self.df[col] = self.df['sell_price'] - self.df[f'sell_price_lag_{lag}']
            self.new_colname.append(col)
            col = f'sell_price_lag_{lag}_div'
            self.df[col] = self.df['sell_price'] / self.df[f'sell_price_lag_{lag}']
            self.new_colname.append(col)


class Price_fe(Feature):
    """
    店舗・商品ごとの価格の基礎統計量
    """
    def create_features(self):
        self.new_colname = ['price_max', 'price_min', 'price_mean', 'price_std', 'price_norm',
                            'price_nunique', 'item_nunique']
        self.df['price_max'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('max').astype(np.float32)
        self.df['price_min'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('min').astype(np.float32)
        self.df['price_mean'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean').astype(np.float32)
        self.df['price_std'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('std').astype(np.float32)
        self.df['price_norm'] = self.df['sell_price'] / self.df['price_max']

        self.df['price_nunique'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
        self.df['item_nunique'] = self.df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.float32)


class Price_StoreItemDate(Feature):
    """
    日ごとの店舗・商品価格の基礎統計量
    ほとんど使える感じではなさそう
    """
    def create_features(self):
        self.new_colname = ['price_store_item_date_max', 'price_store_item_date_min',
                            'price_store_item_date_mean', 'price_store_item_date_std']
        self.df['price_store_item_date_max'] = self.df.groupby(['store_id', 'item_id', 'date'])['sell_price'].transform('max')
        self.df['price_store_item_date_min'] = self.df.groupby(['store_id', 'item_id', 'date'])['sell_price'].transform('min')
        self.df['price_store_item_date_mean'] = self.df.groupby(['store_id', 'item_id', 'date'])['sell_price'].transform('mean')
        self.df['price_store_item_date_std'] = self.df.groupby(['store_id', 'item_id', 'date'])['sell_price'].transform('std')
        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.float32)


class Price_StoreCatDate(Feature):
    """
    日ごとの店舗・商品価格の基礎統計量
    """
    def create_features(self):
        self.new_colname = ['price_store_cat_date_max', 'price_store_cat_date_min',
                            'price_store_cat_date_mean', 'price_store_cat_date_std']
        self.df['price_store_cat_date_max'] = self.df.groupby(['store_id', 'cat_id', 'date'])['sell_price'].transform('max')
        self.df['price_store_cat_date_min'] = self.df.groupby(['store_id', 'cat_id', 'date'])['sell_price'].transform('min')
        self.df['price_store_cat_date_mean'] = self.df.groupby(['store_id', 'cat_id', 'date'])['sell_price'].transform('mean')
        self.df['price_store_cat_date_std'] = self.df.groupby(['store_id', 'cat_id', 'date'])['sell_price'].transform('std')
        for c in self.new_colname:
            self.df[c] = self.df[c].astype(np.float32)


class SalesPeriod(Feature):
    """
    idごとの販売日数
    """
    def create_features(self):
        self.new_colname = ['sales_period']
        _df = self.df.dropna()[['id', 'date']].copy()
        _df = _df.groupby('id').count().reset_index()
        rep = {k:v for k, v in zip(_df['id'].values, _df['date'].values)}
        del _df
        self.df['sales_period'] = self.df['id'].map(rep)


class ReferNotebook(Feature):
    def create_feaures(self):
        self.new_colname = ['item_id', 'dept_id', ]



if __name__ == '__main__':

    save_dir = '../features'

    # Load Data
    # df = load_data(merge=True)

    with open('../data/input/data.pkl', 'rb') as f:
        df = pickle.load(f)

    print(df.head())
    print(df.columns)

    # SellPrice(df, dir=save_dir).run().save()
    # TimeFeatures(df, dir=save_dir).run().save()
    # Snap(df, dir=save_dir).run().save()
    # Lag(df, dir=save_dir).run().save()
    # Lag_RollMean_28(df, dir=save_dir).run().save()
    # Lag_RollMean_45(df, dir=save_dir).run().save()
    # Event(df, dir=save_dir).run().save()
    # Ids(df, dir=save_dir).run().save()
    # Lag_SellPrice(df, dir=save_dir).run().save()
    # Lag_SellPrice_diff(df, dir=save_dir).run().save()
    # Price_fe(df, dir=save_dir).run().save()
    # Price_StoreItemDate(df, dir=save_dir).run().save()
    # Price_StoreCatDate(df, dir=save_dir).run().save()
    # SalesPeriod(df, dir=save_dir).run().save()


