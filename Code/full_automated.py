import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
import os

# Calling arguments
args = sys.argv

path_in = args[1]
data_file = args[2]

path = os.path.dirname(path_in)
data_file = os.path.join(path, data_file)
print(data_file)

tf.keras.backend.set_floatx('float64')

# Script parameters
random_state = 3
seeds = np.arange(100)
#seeds = np.arange(1)

# Read in data, preprocessing
data = pd.read_csv(data_file, header=0, sep=';', decimal=',')
col_names = data.columns
X = data[col_names[3:]]
y = data[col_names[1]]

def scale(train_set, valid_set, feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_set)
    return [pd.DataFrame(scaler.transform(x), columns=x.columns) for x in [train_set, valid_set]] + [scaler]

def get_model(n_input_features, learning_rate=0.01):
    model = keras.models.Sequential([
        keras.layers.GaussianNoise(stddev=0.1, input_shape=(n_input_features, )),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    return model

kf = KFold(n_splits=6, random_state=random_state, shuffle=True)
i = -1

for train_idx, test_idx in kf.split(X):
    i += 1
    X_train, X_valid = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_valid = pd.DataFrame(y.iloc[train_idx]), pd.DataFrame(y.iloc[test_idx])
    X_train_sc, X_valid_sc, _ = scale(X_train, X_valid, (-1, 1))
    y_train_sc, y_valid_sc, scaley = scale(y_train, y_valid, (0, 1))

    skb = SelectKBest(f_regression, k='all')
    skb.fit(X_train_sc, np.ravel(y_train_sc))
    _, anova_sorted = zip(*sorted(zip(skb.scores_, X_train_sc.columns), reverse=True))

    for n_features in [3, 5, 63]:
        cols_anova = anova_sorted[:n_features]

        rfe_selected_cols = []

        for seed in seeds:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True, bootstrap=True, random_state=seed)
            selector = RFE(rf, n_features_to_select=n_features, step=0.1).fit(X_train_sc, np.ravel(y_train_sc))
            rfe_selected_cols.append([X.columns[i] for i in range(len(selector.support_)) if selector.support_[i]])

        vals, counts = np.unique(rfe_selected_cols, axis=0, return_counts=True)
        cols_rfe = vals[np.argmax(counts)]

        all_selected = [c for lst in rfe_selected_cols for c in lst]
        vals_fq, counts_fq = np.unique(all_selected, axis=0, return_counts=True)
        counts_fq, vals_fq = zip(*sorted(zip(counts_fq, vals_fq), reverse=True))
        cols_rfe_fq = vals_fq[:n_features]
        
        results_dict = {
            'i': i,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'n_features': n_features,
            'cols_anova': cols_anova,
            'cols_rfe': cols_rfe,
            'cols_rfe_fq': cols_rfe_fq,
            'vals_rfe': vals,
            'counts_rfe': counts,
            'vals_rfe_fq': vals_fq,
            'counts_rfe_fq': counts_fq,
            'anova_sorted': anova_sorted
        }
        results_path = path + '/feature_analysis_i{}_nfeatures{}.pkl'.format(i, n_features)
        with open(results_path, 'wb+') as f:
            pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

        for model_type in ['anova', 'rfe', 'rfe_fq']:
            if model_type == 'anova':
                feature_list = cols_anova
            elif model_type == 'rfe':
                feature_list = cols_rfe
            else:
                feature_list = cols_rfe_fq
            feature_list = list(feature_list)

            for seed in seeds:
                tf.keras.backend.clear_session()
                tf.random.set_seed(seed)
                model = get_model(n_features)
                history = model.fit(X_train_sc[feature_list], y_train_sc, validation_data=(X_valid_sc[feature_list], y_valid_sc), epochs=25, verbose=0)
                model_path = path + '/model_i{}_nfeatures{}_type_{}_seed{}'.format(i, n_features, model_type, seed)
                model.save(model_path)
                history_path = path + '/history_i{}_nfeatures{}_type_{}_seed{}.pkl'.format(i, n_features, model_type, seed)
                with open(history_path, 'wb+') as f:
                    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)