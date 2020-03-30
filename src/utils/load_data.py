import pickle, os

def load_data(data_dir):
    # Load Data
    with open(os.path.join(data_dir, 'prep_train.pkl'), 'rb') as f:
        train = pickle.load(f)

    with open(os.path.join(data_dir, 'validation.pkl'), 'rb') as f:
        vals = pickle.load(f)

    with open(os.path.join(data_dir, 'evaluation.pkl'), 'rb') as f:
        evals = pickle.load(f)

    return train, vals, evals