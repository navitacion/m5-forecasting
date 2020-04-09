import pandas as pd
import numpy as np
from utils.utils import load_data
import pickle

with open('../data/input/data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.columns)

data = data[['id', 'date', 'part', 'demand', 'sell_price']]
data['sell_price'] = data['sell_price'].astype(np.float32)
data.to_feather('../features/SellPrice.ftr')
