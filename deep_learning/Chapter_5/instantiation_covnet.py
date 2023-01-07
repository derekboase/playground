from keras.utils import to_categorical
from keras.datasets import mnist
from keras import layers, models
from os import name, system

    # Clear the output of the VS code compiler
if name == 'posix':
    system('clear')
elif name == 'nt':
    system('cls')

    # Import and reshape the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # Reshape the images
train_images = train_images.reshape(60000, 28, 28, 1) 
test_images = test_images.reshape(10000, 28, 28, 1) 
        
        # Normalizing the images
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
        
        # One hot encoding of the test labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels) 

    # Model instantiation
model = models.Sequential()
# What is the channel? Why 32? 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

    # Model compilation
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Model fitting
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# model.summary() # Uncomment to view the model summary 