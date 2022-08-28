import time
import os 
clear = lambda: os.system('cls')
clear()

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import reuters
from keras import layers, losses, metrics, models, optimizers

VAL_SET = 1000
MAX_WORDS = 10000

def vectorize_sequences(data, dimension=MAX_WORDS):
    results = np.zeros((len(data), dimension)) # Make a list of len(sequence) samples each with MAX_WORDS one-hot encoding
    for i, sequence in enumerate(data):
        results[i, sequence] = 1
    return results

# Data import
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=MAX_WORDS)
start_time = time.time_ns()

# Data pre-processing
    # Data formatting
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = vectorize_sequences(train_labels, dimension=46)
y_test = vectorize_sequences(test_labels, dimension=46)
    # Validation segment
partial_x_train = x_train[VAL_SET:]
x_val = x_train[:VAL_SET]
partial_y_train = y_train[VAL_SET:]
y_val = y_train[:VAL_SET]


# Model definition
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(MAX_WORDS, )))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Network
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=128, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
print(results)

# Analytics 
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(history.history['accuracy']) + 1)

print(f'The total runtime is: {(time.time_ns() - start_time)*1e-9}') 
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# plt.show()
