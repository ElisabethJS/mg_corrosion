import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow import keras
import pickle

tf.keras.backend.set_floatx('float64')

# Script parameters
random_state = 42
n_runs = 100
seeds = np.arange(n_runs)
path = '..\\Data\\'
csv_path = path + 'ze41_mol_desc_db_red.csv'
res_path = path + 'rfe_results_recalc.pkl'

univ_regr_sel = ['CATS2D_03_AP', 'CATS3D_03_AP', 'CATS3D_02_AP', 'LUMO / eV', 'P_VSA_MR_5', 'P_VSA_LogP_2', 'CATS2D_02_PN', 'CATS2D_03_DP', 'H3m',
    'nRNH2', 'H%', 'Mor14u', 'CATS2D_03_AN', 'Eta_epsi_5', 'CATS3D_03_DP', 'P_VSA_e_3', 'Mor19m', 'Mv', 'Mor12e', 'B02[C-N]', 'Eta_F_A', 'R2e+',
    'Mor14s', 'NssCH2', 'Mor32m', 'SHED_AP', 'P_VSA_ppp_con', 'Mor14v', 'Mor08s', 'E2m', 'SPH', 'MATS7m', 'F03[N-O]', 'SsNH2', 'CATS3D_04_NL',
    'E3p', 'Hlgap / eV', 'CATS2D_04_AA', 'Mor08i', 'HOMO / eV', 'TDB03e', 'TDB02e', 'B03[N-O]', 'CATS3D_02_AN', 'Mi', 'Mor27m', 'TDB04v', 'ZM2V',
    'Mor12p', 'Eta_FL_A', 'GATS1m', 'Ui', 'Mor12m', 'SHED_PN', 'SHED_DP', 'GATS1v', 'MLOGP', 'MPC07', 'nN', 'DLS_05', 'nArCOOH', 'Mor08p',
    'SpPosA_B(e)']

rfe_sel = ['P_VSA_MR_5', 'LUMO / eV', 'Mor04m', 'Mor22s', 'E1p', 'P_VSA_LogP_2', 'HOMO / eV', 'MATS5v', 'Mor14s', 'Mor29v', 'Mor14u', 'GATS5v', 'GATS2s', 'MATS5m', 'Mor32m',
        'H3m', 'CATS3D_02_AP', 'TDB04s', 'R2e+', 'E2s', 'R5p+', 'ISH', 'DISPm', 'R5i+', 'Ds', 'Mor04i', 'E2m', 'Mor28s', 'TDB03m', 'Mor19m', 'Mor11u', 'VE2sign_G',
        'Mor03s', 'SpMAD_RG', 'E2v', 'R3s+', 'R5e+', 'R2u+', 'Mor15i', 'H0v', 'T(N..O)', 'E1i', 'Eta_epsi_5', 'E3e', 'MATS4s', 'Mor13u', 'H1p', 'X4Av', 'Mor15s', 'Hy',
        'HATS0p', 'Eig03_AEA(dm)', 'X3Av', 'VE1sign_G', 'GATS5m', 'E2e', 'Mor10e', 'MATS8p', 'TDB01m', 'GATS4s', 'TDB04m', 'PJI3', 'Mor16m']

# Read in data, preprocessing
data = pd.read_csv(csv_path, header=0, sep=';', decimal=',')
col_names = data.columns
X = data[col_names[3:]]
y = data[col_names[1]]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=random_state)
[X_train, X_valid, y_train, y_valid] = [pd.DataFrame(x) for x in [X_train, X_valid, y_train, y_valid]]

scalex = MinMaxScaler(feature_range=(-1,1))
scalex.fit(X_train)
[X_train_sc, X_valid_sc] = [pd.DataFrame(scalex.transform(x), columns=X.columns) for x in [X_train, X_valid]]

scaley = MinMaxScaler(feature_range=(0, 1))
scaley.fit(y_train)
[y_train_sc, y_valid_sc] = [pd.DataFrame(scaley.transform(y), columns=y.columns) for y in [y_train, y_valid]]


# Train model multiple times on the best performing set
def get_model(n_input_features):
    model = keras.models.Sequential([
        keras.layers.GaussianNoise(stddev=0.1),
        keras.layers.Dense(100, activation='relu', input_shape=(n_input_features,)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
        ])
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.005),
        loss='mean_squared_error')
    return model

def get_x_cols(n_features, col_source):
    if col_source == 'urs':
        return univ_regr_sel[:n_features]
    return rfe_sel[:n_features]

results = {}

for n_features in [3, 5, 63]:
    for col_source in ['rfe']:
        losses = []
        predictions = pd.DataFrame(y_valid)
        for seed in seeds:
            print(n_features, col_source, seed)
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed)
            model = get_model(n_features)
            x_cols = get_x_cols(n_runs, col_source)
            history = model.fit(X_train_sc[x_cols], y_train_sc, validation_data=(X_valid_sc[x_cols], y_valid_sc), epochs=25, verbose=0)
            losses.append(history.history['val_loss'][-1])
            y_pred = model.predict(tf.convert_to_tensor(X_valid_sc[x_cols]))
            predictions[seed] = scaley.inverse_transform(y_pred)
        predictions = predictions.drop(labels='inhibition efficiency ZE41 / %', axis=1)
        results[(n_features, col_source)] = {'loss mean': np.mean(losses), 'loss std': np.std(losses), 'pred mean': predictions.mean(axis=1),
            'pred std': predictions.std(axis=1)}

# save results
with open(res_path, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
