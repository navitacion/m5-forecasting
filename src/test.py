import gc, os, pickle, datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.utils import reduce_mem_usage

with open('../data/input/data.pkl', 'rb') as f:
    df = pickle.load(f)

df = df[df['store_id'] == 'CA_1']

print(df.head())
print(df.index)


temp = df['sell_price'].values

temp[[1, 3]] = [300, 300]

print(temp)
