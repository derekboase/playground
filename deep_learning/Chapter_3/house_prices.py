import time
import os
from tracemalloc import start 
clear = lambda: os.system('cls')
clear()

import matplotlib.pyplot as plt
import numpy as np
from  keras import layers, models
from keras.datasets import boston_housing

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Data import 
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

start_time = time.time_ns()
# Data pre-processing 
train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)
train_data = (train_data - train_mean)/train_std
test_data = (test_data - train_mean)/train_std

# K-fold validation and iteration
k = 4
num_val_samples = len(train_data) // k
num_epochs = 50
all_mae_histories = []
for i in range(k):
    print(f'\nProcessing Fold #{i}')
    lower_bound, upper_bound = i*num_val_samples, (i + 1)*num_val_samples
    val_data = train_data[lower_bound:upper_bound]
    val_targets = train_targets[lower_bound:upper_bound]

    partial_train_data = np.concatenate([train_data[:lower_bound], train_data[upper_bound:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:lower_bound], train_targets[upper_bound:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, batch_size=1, epochs=num_epochs, validation_data=(val_data, val_targets))
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

# Averaging the MAEs
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plotting
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

print(f'The total run time of the program is: {(time.time_ns() - start_time)*1e-9}')
