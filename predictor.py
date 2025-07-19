import numpy as np
import pandas as pd
from sklearn import preprocessing

#from tensorflow import keras
#from tensorflow.keras import layers

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

import pickle
data = pd.read_csv("solar_flare.csv")
date = np.array(pd.DataFrame(data, columns=['date in number format']))
mean_mag_field = np.array(pd.DataFrame(data, columns=['mean_mag_field']))
solar_irradiance = np.array(pd.DataFrame(data, columns=['total_solar_irradiance']))
radio_flux_density = np.array(pd.DataFrame(data, columns=['radio_flux_density']))
sunspot_number = np.array(pd.DataFrame(data, columns=['sunspot_number']))
sunspot_area = np.array(pd.DataFrame(data, columns=['sunspot_area_10e-6hemis']))
xray = np.array(pd.DataFrame(data, columns=['xray_flux']))
lyman_alpha = np.array(pd.DataFrame(data, columns=['lyman_alpha']))
flare_c = np.array(pd.DataFrame(data, columns=['flare_c']))
flare_m = np.array(pd.DataFrame(data, columns=['flare_m']))
flare_x = np.array(pd.DataFrame(data, columns=['flare_x']))
flare_s = np.array(pd.DataFrame(data, columns=['flare_s']))
total_flares = np.array(pd.DataFrame(data, columns=['Flare_total']))


def normalize_output(y_train, y_min, y_max):
    y_train = (y_train - y_min) / (y_max - y_min)
    return y_train


date = normalize_output(date, np.amin(date), np.amax(date))
mean_mag_field = normalize_output(mean_mag_field, np.amin(mean_mag_field), np.amax(mean_mag_field))
solar_irradiance = normalize_output(solar_irradiance, np.amin(solar_irradiance), np.amax(solar_irradiance))
radio_flux_density = normalize_output(radio_flux_density, np.amin(radio_flux_density), np.amax(radio_flux_density))
sunspot_area = normalize_output(sunspot_area, np.amin(sunspot_area), np.amax(sunspot_area))
sunspot_number = normalize_output(sunspot_number, np.amin(sunspot_number), np.amax(sunspot_number))
xray = np.array(pd.DataFrame(data, columns=['xray_flux']))
lyman_alpha = np.array(pd.DataFrame(data, columns=['lyman_alpha']))

flare_c = normalize_output(flare_c, np.amin(flare_c), np.amax(flare_c))
flare_m = normalize_output(flare_m, np.amin(flare_m), np.amax(flare_m))
flare_x = normalize_output(flare_x, np.amin(flare_x), np.amax(flare_x))
flare_s = normalize_output(flare_s, np.amin(flare_s), np.amax(flare_s))

conso1 = np.concatenate((date, mean_mag_field, solar_irradiance), axis=1)
conso2 = np.concatenate(
    (radio_flux_density, sunspot_area, sunspot_number), axis=1)

X = np.concatenate((conso1, conso2), axis=1)
Y = np.concatenate((flare_c, flare_m, flare_x, flare_s), axis=1)


X_train = X[:5476]
X_test = X[5477:-2]
print(np.shape(X_test))
X_train = np.reshape(X_train, (2738, 12))
X_test = np.reshape(X_test, (1323, 12))
Y_train = Y[:5476]
Y_test = Y[5477:-2]
Y_train = np.reshape(Y_train, (2738, 8))
Y_test = np.reshape(Y_test, (1323, 8))


def build_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=[12]),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(128, activation='relu'),
        Dense(8, activation='relu')
    ])
    opt = optimizers.SGD(learning_rate=0.01, nesterov=True)
    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
model = build_model()
model.summary()
history = model.fit(X_train, Y_train, validation_split=0.2,
                    epochs=3000, batch_size=100, callbacks=[es])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.yscale('log')
plt.legend(['loss', 'val_loss'], loc='upper right', prop={'size': 20})
plt.show()

Y_pred = model.predict(X_test)
error = Y_pred - Y_test


def denormalise(param, min, max):
    denorm_param = param*(max-min)+min
    return denorm_param


error = np.reshape(error, (2646, 4))

c = error[:, 0]
m = error[:, 1]
x = error[:, 2]
s = error[:, 3]

c = denormalise(c, np.amin(c), np.amax(c))
m = denormalise(m, np.amin(m), np.amax(m))
x = denormalise(x, np.amin(x), np.amax(x))
s = denormalise(s, np.amin(s), np.amax(s))

plt.hist(np.abs(c), 100)
plt.hist(np.abs(m), 100)
plt.hist(np.abs(x), 100)
plt.hist(np.abs(s), 100)

with open('test.pkl', 'wb') as f:
    pickle.dump(Y_pred, f)
