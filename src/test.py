import gc, os, pickle, datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.utils import reduce_mem_usage

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

h = 28
max_lags = 70
tr_last = 1913
fday = datetime.datetime(2016, 4, 25)

def create_dt(is_train=True, nrows=None, first_day=1200):
    prices = pd.read_csv("..data/input/sell_prices.csv", dtype=PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    cal = pd.read_csv("../data/input/calendar.csv", dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    start_day = max(1 if is_train else tr_last - max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("../data/input/sales_train_validation.csv",
                     nrows=nrows, usecols=catcols + numcols, dtype=dtype)

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    if not is_train:
        for day in range(tr_last + 1, tr_last + 28 + 1):
            dt[f"d_{day}"] = np.nan

    dt = pd.melt(dt,
                 id_vars=catcols,
                 value_vars=[col for col in dt.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    return dt
