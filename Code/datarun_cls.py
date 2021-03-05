import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow import keras
import pickle

tf.keras.backend.set_floatx('float64')

# Script parameters
random_state = 81
n_features_to_select = 5
n_runs = 100
seeds = np.arange(n_runs)
path = '..\\Data\\'
csv_path = path + 'ze41_mol_desc_db_red.csv'
df_path = path + 'predictions_cls_seed{}_feat{}_runs{}.csv'.format(random_state, n_features_to_select, n_runs)
rfe_path = path + 'rfe_cls_res_seed{}_feat{}_runs{}.pkl'.format(random_state, n_features_to_select, n_runs)

# Read in data, preprocessing
data = pd.read_csv(csv_path, header=0, sep=';', decimal=',')
col_names = data.columns
X = data[col_names[3:]]
y = data[col_names[1]]

def get_class(x):
    if x < -40:
        return 0
    if x < 0:
        return 1
    if x < 40:
        return 2
    return 3

y = y.apply(get_class)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=random_state)
[X_train, X_valid, y_train, y_valid] = [pd.DataFrame(x) for x in [X_train, X_valid, y_train, y_valid]]

scalex = MinMaxScaler(feature_range=(-1,1))
scalex.fit(X_train)
[X_train_sc, X_valid_sc] = [pd.DataFrame(scalex.transform(x), columns=X.columns) for x in [X_train, X_valid]]

# RFE
selected_cols = []

for seed in seeds:
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, bootstrap=True, random_state=seed)
    selector = RFE(rf, n_features_to_select=n_features_to_select, step=0.1).fit(X_train_sc, np.ravel(y_train))
    selected_cols.append([X.columns[i] for i in range(len(selector.support_)) if selector.support_[i]])
    if seed%10 == 9:
        print('RFE {:.2f}% done'.format((seed+1)*100/len(seeds)))

vals, counts = np.unique(selected_cols, axis=0, return_counts=True)
best_features = vals[np.argmax(counts)]

# Train model multiple times on the best performing set
def get_model(n_input_features):
    model = keras.models.Sequential([
        keras.layers.GaussianNoise(stddev=0.1),
        keras.layers.Dense(50, activation='relu', input_shape=(n_input_features,)),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
        ])
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    return model

accs = []
predictions = y_valid.copy()

for seed in seeds:
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = get_model(n_features_to_select)
    history = model.fit(X_train_sc[best_features], y_train, validation_data=(X_valid_sc[best_features], y_valid), epochs=25, verbose=0)
    accs.append(history.history['val_accuracy'][-1])
    y_pred = model.predict(tf.convert_to_tensor(X_valid_sc[best_features]))
    predictions[seed] = np.argmax(y_pred, axis=1)
    if seed%10 == 9:
        print('Training {:.2f}% done'.format((seed+1)*100/len(seeds)))

df_a = pd.DataFrame([0] + accs, columns=['accuracy']).T
df_a.columns = predictions.columns
predictions = predictions.append(df_a).T

# save results
predictions.to_csv(df_path, header=True, sep=';', decimal=',')
with open(rfe_path, 'wb') as f:
    pickle.dump({'vals': vals, 'counts': counts}, f, pickle.HIGHEST_PROTOCOL)

