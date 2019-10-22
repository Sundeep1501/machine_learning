import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Test data size " + str(x_train.shape))
print(y_train.shape)
print("Train test data size " + str(x_test.shape))
print(y_test.shape)


# check image data format. TensorFlow has channels_last format
img_data_format = keras.backend.image_data_format()
print("Backend image data format is " + img_data_format)

# black-white image has channel 1 and color image has channel 3 (RGB)
image_channel_size = 1

sample = x_train[0].shape

# reshape the train and test input data with channels
if(img_data_format == 'channels_last'):
    x_train = x_train.reshape(x_train.shape[0], sample[0], sample[1], image_channel_size)
    x_test = x_test.reshape(x_test.shape[0], sample[0], sample[1], image_channel_size)
    input_img_shape = (sample[0], sample[1], image_channel_size)
else:
    x_train = x_train.reshape(x_train.shape[0], image_channel_size, sample[0], sample[1])
    x_test = x_test.reshape(x_test.shape[0], image_channel_size, sample[0], sample[1])
    input_img_shape = (image_channel_size, sample[0], sample[1])

print("Input shape with channel " + str(input_img_shape))

# convert out put single dimension values to categorical binary vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Build CNN (sequence of layers)
model = Sequential()

# Add Convolution layer
model.add(Conv2D(32, # number of features
                 kernel_size = (3, 3), # convolution filter size
                 input_shape = input_img_shape,
                 activation = 'relu'
                 ))

# Add MaxPooling layer to decrese the size of the feature maps
model.add(MaxPooling2D(pool_size = (2, 2)))

#EXTRA CONVOLUTION LAYER TO IMPROVE ACCURACY
model.add(Conv2D(32,
                 kernel_size = (3, 3),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add Flatten layer to convert 2d to vector
model.add(Flatten())

# Add Hidden layer (fully connected layer)
model.add(Dense(128, activation='relu'))

# Add Output layer (fully connected layer)
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=12,
          verbose=2,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
