from scipy.io import loadmat
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os.path

x = loadmat('Salinas_corrected.mat')
data = x['salinas_corrected']
print(data.shape)
data = data[0:,0:,4:]
print(data.shape)

gx = loadmat('Salinas_gt.mat')
gt = gx['salinas_gt']
print(gt.shape)

classes = np.zeros(17)

for i in range(512):
    for j in range(217):
        classes[gt[i][j]] = classes[gt[i][j]] + 1

print(classes)

# READ PIXEL DATA ALL BANDS
X = np.empty((54129, 200))
Y = np.array([])

index = 0

for i in range(512):
    for j in range(217):
        # remove background pixels
        if gt[i][j] != 0:
            X[index] = np.array(data[i][j])
            Y = np.append(Y, gt[i][j]-1)
            index = index+1

Y = to_categorical(Y)
print(X.shape)
print(Y.shape)

# SPLIT THE DATA INTO TRAINING SET AND TEST SET
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print("\nTraining and Test Data")

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



# <<<<<<<<<< PART 2 >>>>>>>>>>>
# Apply CNN model for the Dataset
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
import keras.backend as K

model = Sequential()

# Add Conv1D layer which takes band vector of each pixel
model.add(Conv1D(filters=64,
                 kernel_size=10,
                 input_shape=(X_train.shape[1],1),
                 activation='relu'))

# Add MaxPooling layer to decrese the size of the feature maps
model.add(MaxPooling1D(pool_size = 2))

# Add Flatten layer to convert 2d to vector
model.add(Flatten())

# Add Hidden layer (fully connected layer)
model.add(Dense(512, activation='relu'))

# Add Output layer (fully connected layer)
model.add(Dense(Y_train[0].shape[0], activation='softmax'))

# Compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
            batch_size=32,
            epochs=200)
model.save('salinas.h5')