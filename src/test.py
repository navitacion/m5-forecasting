import pickle
import pandas as pd

# target_path = ['../features/Weekday.ftr', '../features/Snap.ftr', '../features/Lag.ftr',
#                '../features/RollMean.ftr', '../features/SellPrice.ftr', '../features/Snap.ftr',
#                '../features/Weekday.ftr']
#
# for path in target_path:
#     d = pd.read_feather(path)
#     print(path)
#     print(d.columns)
#     print('#'*30)

a = pd.read_csv('../data/output/LightGBM_01_rmse_1.875.csv')

print(a.head())
print(a.shape)
