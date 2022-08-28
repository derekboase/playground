import time
import os 
clear = lambda: os.system('cls')
clear()

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import imdb
from keras import layers, losses, metrics, models, optimizers

MAX_WORDS = 10000

def vectorize_sequences(sequences, dimension=MAX_WORDS):
    results = np.zeros((len(sequences), dimension)) # Make a list of len(sequence) samples each with MAX_WORDS one-hot encoding
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Data import
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS)
start_time = time.time_ns()

# Data pre-processing
    # Data formatting
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
    # Validation segment
partial_x_train = x_train[10000:]
x_val = x_train[:10000]
partial_y_train = y_train[10000:]
y_val = y_train[:10000]


# Model definition
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(MAX_WORDS, )))
model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

# Network
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
print(results)

# Analytics 
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']
epochs = range(1, len(history_dict['binary_accuracy']) + 1)

print(f'The total runtime is: {(time.time_ns() - start_time)*1e-9}')
 
# Plotting 
plt.figure(1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
