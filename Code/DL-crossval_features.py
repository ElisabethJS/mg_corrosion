import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import linregress
import tensorflow as tf
from tensorflow import keras
import pickle

tf.keras.backend.set_floatx('float64')

# Script parameters
random_state = 42
seeds = np.arange(100)

path = '..\\Data\\'
csv_path = path + 'ze41_mol_desc_db_red.csv'
stats_path = path + 'cv_cleaned.csv'
#anova_path = path + 'anova_cv.pkl'

# col names
anova = ['CATS2D_03_AP', 'CATS3D_03_AP', 'CATS3D_02_AP', 'LUMO / eV',
       'P_VSA_MR_5', 'P_VSA_LogP_2', 'CATS2D_02_PN', 'CATS2D_03_DP', 'H3m',
       'nRNH2', 'H%', 'Mor14u', 'CATS2D_03_AN', 'Eta_epsi_5', 'CATS3D_03_DP',
       'P_VSA_e_3', 'Mor19m', 'Mv', 'Mor12e', 'B02[C-N]', 'Eta_F_A', 'R2e+',
       'Mor14s', 'NssCH2', 'Mor32m', 'SHED_AP', 'P_VSA_ppp_con', 'Mor14v',
       'Mor08s', 'E2m', 'SPH', 'MATS7m', 'F03[N-O]', 'SsNH2', 'CATS3D_04_NL',
       'E3p', 'Hlgap / eV', 'CATS2D_04_AA', 'Mor08i', 'HOMO / eV', 'TDB03e',
       'TDB02e', 'B03[N-O]', 'CATS3D_02_AN', 'Mi', 'Mor27m', 'TDB04v', 'ZM2V',
       'Mor12p', 'Eta_FL_A', 'GATS1m', 'Ui', 'Mor12m', 'SHED_PN', 'SHED_DP',
       'GATS1v', 'MLOGP', 'MPC07', 'nN', 'DLS_05', 'nArCOOH', 'Mor08p',
       'SpPosA_B(e)']
rfe = ['P_VSA_MR_5', 'Mor22s', 'Mor04m', 'LUMO / eV', 'E1p', 'HOMO / eV', 'P_VSA_LogP_2', 'Mor29v', 'MATS5v',
           'Mor14s', 'Mor14u', 'CATS3D_02_AP', 'GATS5v', 'MATS5m', 'GATS2s', 'Mor32m', 'H3m', 'TDB04s', 'E2s', 'R5p+',
           'R2e+', 'ISH', 'DISPm', 'R5i+', 'Mor04i', 'Ds', 'Mor03s', 'E2m', 'Mor28s', 'Mor11u', 'TDB03m', 'Mor19m',
           'VE2sign_G', 'SpMAD_RG', 'R3s+', 'R5e+', 'E2v', 'Mor15i', 'T(N..O)', 'R2u+', 'MATS8p', 'Eta_epsi_5',
           'MATS4s', 'H0v', 'Hy', 'E1i', 'VE1sign_G', 'Mor15s', 'E3e', 'Mor13u', 'Eig03_AEA(dm)', 'X4Av', 'P_VSA_e_3',
           'Mor29e', 'Mor16m', 'GATS5m', 'E3p', 'E2e', 'X3Av', 'Mor19u', 'GATS4s', 'E3v', 'TDB04m']

# Read in data, preprocessing
data = pd.read_csv(csv_path, header=0, sep=';', decimal=',')
col_names = data.columns
X = data[col_names[3:]]
y = data[col_names[1]]

def scale_x(X_train, X_valid):
    scalex = MinMaxScaler(feature_range=(-1, 1))
    scalex.fit(X_train)
    return [pd.DataFrame(scalex.transform(x), columns=X.columns) for x in [X_train, X_valid]]

def scale_y(y_train, y_valid):
    scaley = MinMaxScaler(feature_range=(0, 1))
    scaley.fit(y_train)
    return [pd.DataFrame(scaley.transform(y), columns=y.columns) for y in [y_train, y_valid]] + [scaley]

# Train model multiple times on the best performing set
def get_model(n_input_features):
    model = keras.models.Sequential([
        keras.layers.GaussianNoise(stddev=0.1, input_shape=(n_input_features,)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
        ])
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='mean_squared_error')
    return model

def rmse(x, y):
    return np.sqrt(((x-y)**2).mean())

def evaluate(predictions, y_valid):
    means = predictions[predictions.columns[1:]].mean(axis=1).to_numpy()
    yv = y_valid[col_names[1]].to_numpy()
    r, p = linregress(means, yv)[2:4]
    return [r**2, rmse(means, yv), r, p]

kf = KFold(n_splits=10, random_state=random_state, shuffle=True)

stats_all_folds = []
feat_anova = []
i = 0

for train_index, test_index in kf.split(X):
    #if 13 in train_index: train_index.remove(13)
    #if 13 in test_index: test_index.remove(13)
    X_train, X_valid = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_valid = pd.DataFrame(y.iloc[train_index]), pd.DataFrame(y.iloc[test_index])
    X_train_sc, X_valid_sc = scale_x(X_train, X_valid)
    y_train_sc, y_valid_sc, scaley = scale_y(y_train, y_valid)
   
    for n_features in [3, 5, 63]:#, 1260]:
        skb = SelectKBest(f_regression, k=n_features)
        _ = skb.fit_transform(X_train_sc, np.ravel(y_train_sc))
        _, cls1 = zip(*sorted(zip(skb.scores_, X_train_sc.columns), reverse=True))
        feat_anova.append(cls1[:n_features])
        cols = list(cls1[:n_features])
        
        for ty in ['a']:#, 'b']:
            print(i, n_features, ty)
            #if n_features == 1260:
            #    cols = col_names[3:]
            #    if ty == 'b':
            #        continue
            #elif ty == 'a'
            #    cols = anova[:n_features]
            #else:
            #    cols = rfe[:n_features]
        
            predictions = pd.DataFrame(y_valid)
            
            X_train_sel = X_train_sc[cols]
            X_val_sel = X_valid_sc[cols]
            
            for seed in seeds:
                tf.keras.backend.clear_session()
                tf.random.set_seed(seed)
                model = get_model(n_features)
                history = model.fit(X_train_sel, y_train_sc, validation_data=(X_val_sel, y_valid), epochs=25, verbose=0)
                y_pred = model.predict(tf.convert_to_tensor(X_val_sel))
                predictions[seed] = scaley.inverse_transform(y_pred)

            evls = evaluate(predictions, y_valid)
            stats_all_folds.append([i, n_features, ty] + evls)
    i += 1

stats_all_folds = pd.DataFrame(stats_all_folds, columns=['Index', 'Features', 'Type', 'R^2', 'RMSE', 'r', 'p'])

# save results
stats_all_folds.to_csv(stats_path, header=True, sep=';', decimal=',')
with open(anova_path, 'wb') as f:
    pickle.dump(feat_anova, f, pickle.HIGHEST_PROTOCOL)