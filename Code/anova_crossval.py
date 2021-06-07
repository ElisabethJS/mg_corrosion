import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow import keras
from scipy.stats import linregress
import pickle

tf.keras.backend.set_floatx('float64')

# Script parameters
random_state = 42
n_features_to_select = 5
n_runs = 100
seeds = np.arange(n_runs)
path = '..\\Data\\'
csv_path = path + 'ze41_mol_desc_db_red.csv'
df_path = path + 'predictions_seed{}_feat{}_runs{}.csv'.format(random_state, n_features_to_select, n_runs)
rfe_path = path + 'rfe_res_seed{}_feat{}_runs{}.pkl'.format(random_state, n_features_to_select, n_runs)

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
        keras.layers.GaussianNoise(stddev=0.1),
        keras.layers.Dense(50, activation='relu', input_shape=(n_input_features,)),
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
    means = predictions[predictions.columns[1:]].means(axis=1).to_numpy()
    yv = y_valid[col_names[1]].to_numpy()
    r, p = linregress(means, yv)[2:4]
    return [r**2, rmse(means, yv), r, p]

kf = KFold(n_splits=10, random_state=random_state, shuffle=True)


# RFE
selected_cols = []

for seed in seeds:
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True, bootstrap=True, random_state=seed)
    selector = RFE(rf, n_features_to_select=n_features_to_select, step=0.1).fit(X_train_sc, np.ravel(y_train_sc))
    selected_cols.append([X.columns[i] for i in range(len(selector.support_)) if selector.support_[i]])
    if seed%10 == 9:
        print('RFE {:.2f}% done'.format((seed+1)*100/len(seeds)))

vals, counts = np.unique(selected_cols, axis=0, return_counts=True)
best_features = vals[np.argmax(counts)]



losses = []
predictions = pd.DataFrame(y_valid)

for seed in seeds:
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = get_model(n_features_to_select)
    history = model.fit(X_train_sc[best_features], y_train_sc, validation_data=(X_valid_sc[best_features], y_valid_sc), epochs=25, verbose=0)
    losses.append(history.history['val_loss'][-1])
    y_pred = model.predict(tf.convert_to_tensor(X_valid_sc[best_features]))
    predictions[seed] = scaley.inverse_transform(y_pred)
    if seed%10 == 9:
        print('Training {:.2f}% done'.format((seed+1)*100/len(seeds)))

df_l = pd.DataFrame([0] + losses, columns=['loss']).T
df_l.columns = predictions.columns
predictions = predictions.append(df_l).T

# save results
predictions.to_csv(df_path, header=True, sep=';', decimal=',')
with open(rfe_path, 'wb') as f:
    pickle.dump({'vals': vals, 'counts': counts}, f, pickle.HIGHEST_PROTOCOL)

