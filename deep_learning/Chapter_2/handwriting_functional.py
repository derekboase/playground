import time
import os 
clear = lambda: os.system('cls')
clear()

from keras.utils import to_categorical
from keras.datasets import mnist
from keras import layers, models

# Data import
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data pre-processing
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Network configuration
start_time = time.time_ns()
print(start_time)

input_tensor = layers.Input(shape=(784, ))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
network = models.Model(inputs=input_tensor, outputs=output_tensor)

# Network fitting 
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluation
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f'The test accuracy was: {test_acc}')
end_time = time.time_ns()
print(f'The end time is {end_time}\n\nThe total time is {(end_time - start_time)*1e-9}')
