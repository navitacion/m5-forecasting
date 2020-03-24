


def preprocessing(train, vals, evals):
    # delete columns
    del_col = ['wm_yr_wk',
               'event_name_1', 'event_type_1',
               'event_name_2', 'event_type_2',
               'snap_CA', 'snap_TX', 'snap_WI'
               ]
    train.drop(del_col, axis=1, inplace=True)
    vals.drop(del_col, axis=1, inplace=True)
    evals.drop(del_col, axis=1, inplace=True)

    # Prep Date
    _max_year = evals['year'].max()
    train['year_diff'] = _max_year - train['year']
    vals['year_diff'] = _max_year - train['year']
    evals['year_diff'] = _max_year - train['year']

    # Item Category
    train['item_type'] = train['item_id'].apply(lambda x: x.split('_')[0])
    vals['item_type'] = vals['item_id'].apply(lambda x: x.split('_')[0])
    evals['item_type'] = evals['item_id'].apply(lambda x: x.split('_')[0])

    # LabelEncoder
    cols = ['store_id', 'item_id', 'weekday', 'item_type']
    for c in cols:
        lbl = LabelEncoder()
        train[c] = lbl.fit_transform(train[c].values)
        vals[c] = lbl.transform(vals[c].values)
        evals[c] = lbl.transform(evals[c].values)

    # Set Category
    add_cols = ['month', 'year', 'snap']
    cols.extend(add_cols)
    for c in cols:
        train[c] = train[c].astype('category')
        vals[c] = vals[c].astype('category')
        evals[c] = evals[c].astype('category')

    # Sort by Date
    train.sort_values(by='date', ascending=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    del train['date'], vals['date'], evals['date']

    return train, vals, evals