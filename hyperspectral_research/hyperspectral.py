# <<<<<<<<<<< PART 1 >>>>>>>>>>>
# DATE PREPROCESSING 

# IMPORT LIBRARIES
from spectral import *
import numpy as np

import keras
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# LOAD HYPERSPECTRAL AND GROUND TRUTH IMAGE
img = open_image('92AV3C.lan')
print(img)
print('\n')
gt = open_image('92AV3GT.GIS')
print(gt)

# COVNVERT TO NUMPY ARRAY
mm = img.open_memmap()
gtmm = gt.open_memmap()

# DATA CLEANING
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
# We have also reduced the number of bands to 200 by
# removing bands covering the region of water absorption:
# [104-108], [150-163], 220.
mm = np.delete(mm,119,axis=2)
mm = np.delete(mm,np.s_[149:163],axis=2)
mm = np.delete(mm,np.s_[103:108],axis=2)

# READ PIXEL DATA ALL BANDS
X = np.empty((10366, 200))
Y = np.array([])

index = 0
classes = np.zeros(16)

for i in range(145):
    for j in range(145):
        # remove background pixels
        if gtmm[i][j][0] == 0:
            continue
        X[index] = np.array(mm[i][j])
        gtClass = gtmm[i][j][0] - 1
        Y = np.append(Y, gtClass)
        classes[gtClass] = classes[gtClass]+1
        index = index+1

print("\nData samples available each class")
print(classes)


# FROM SINGLE GROUND TRUTH VALUES, CONVERT TO CATEGORICAL
Y = to_categorical(Y)

print("\nComplete Raw Data Shape")
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
              optimizer=keras.optimizers.Adam(),# keras.optimizers.RMSprop()
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=32,
          epochs=50)

print(model.summary())
layer = model.get_layer(index=0)
weights = layer.get_weights()
weights = np.array(weights)
weights = weights[0]
weights = np.transpose(weights)
print(weights)
print(weights.shape)
print(np.transpose(weights[0]).shape)

import matplotlib.pyplot as plt
for x in weights:
    plt.plot(np.transpose(x))
plt.ylabel('some numbers')
plt.show()


y_pred = model.predict_classes(X_test)

target_labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(np.argmax(Y_test,axis=1),y_pred,target_names=target_labels))
cm = confusion_matrix(np.argmax(Y_test,axis=1),y_pred)
print(cm)