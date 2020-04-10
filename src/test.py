from utils.utils import load_data, load_from_feather
import pickle

# Load Data  #####################################
print('Data Loading...')
# From Original
data_dir = '../data/input'
df = load_data(nrows=None, merge=True, data_dir=data_dir)

with open('../data/input/data.pkl', 'wb') as f:
    pickle.dump(df, f)
