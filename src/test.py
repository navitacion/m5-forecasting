from utils.utils import load_data, load_from_feather


# Load Data  #####################################
print('Data Loading...')
# From Original
data_dir = '../data/input'
df_org = load_data(nrows=None, merge=True, data_dir=data_dir)

# From Feather
target_path = ['../features/Weekday.ftr', '../features/Snap.ftr', '../features/Lag.ftr',
               '../features/RollMean.ftr', '../features/SellPrice.ftr'
               ]
df = load_from_feather(target_path)


t = (df_org['demand'] != df['demand']).sum()
print(t)
