from scipy.io import loadmat
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os.path


x = loadmat('Indian_pines_corrected.mat')
data = x['indian_pines_corrected']
print(data.shape)

gx = loadmat('Indian_pines_gt.mat')
gt = gx['indian_pines_gt']
print(gt.shape)

classes = np.zeros(17)

# 145*145 = 21025
for i in range(145):
    for j in range(145):
        classes[gt[i][j]] = classes[gt[i][j]] + 1

print(classes)

# READ PIXEL DATA ALL BANDS
# 21025 - 10776 - 46 -28 -20 -93
X = np.empty((10062, 200))
Y = np.array([])

index = 0

for i in range(145):
    for j in range(145):
        # remove background pixels and low samples records
        # 0,1,7,9,16
        cat = gt[i][j]
        if cat == 0 or cat == 1 or cat == 7 or cat == 9 or cat == 16:
            # skip
            continue

        nCat = cat
        if cat == 2:
            # 1
            nCat = 0
        elif cat == 3:
            # 2
            nCat = 1
        elif cat == 4:
            # 3
            nCat = 2
        elif cat == 5:
            # 4
            nCat = 3
        elif cat == 6:
            # 5
            nCat = 4
        elif cat == 8:
            # 6
            nCat = 5
        elif cat == 10:
            # 7
            nCat = 6
        elif cat == 11:
            # 8
            nCat = 7
        elif cat == 12:
            # 9
            nCat = 8
        elif cat == 13:
            # 10
            nCat = 9
        elif cat == 14:
            # 11
            nCat = 10
        elif cat == 15:
            # 12
            nCat = 11
        
        X[index] = np.array(data[i][j])
        index = index+1
        Y = np.append(Y, nCat)


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

np.save('X_train', X_train)
np.save('Y_train', Y_train)
np.save('X_test', X_test)
np.save('Y_test', Y_test)

from x import y

# <<<<<<<<<< PART 2 >>>>>>>>>>>
# Apply CNN model for the Dataset
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
import keras.backend as K


salinas_model = load_model('salinas1.h5')
weights = salinas_model.get_layer(index=0).get_weights()
weights = np.array(weights)
weights = weights[0]
weights = np.transpose(weights)
print(weights.shape)


model = Sequential()

# Add Conv1D layer which takes band vector of each pixel
model.add(Conv1D(filters=64,
                kernel_size=10,
                input_shape=(X_train.shape[1],1),
                activation='relu',
                kernel_initializer = keras.initializers.constant(weights),
                trainable=False
                ))

# Add MaxPooling layer to decrese the size of the feature maps
model.add(MaxPooling1D(pool_size = 2))

model.add(Conv1D(filters=64,
                kernel_size=10,
                activation='relu'
                ))

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
              optimizer=keras.optimizers.Adagrad(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
            batch_size=32,
            epochs=200)


y_pred = model.predict_classes(X_test)

target_labels = ["A","B","C","D","E","F","G","H","I","J","K","L"]

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_labels))
cm = confusion_matrix(np.argmax(Y_test, axis=1), y_pred)
print(cm)
model.save('indian_pines_6.h5')

# model = load_model('my_model.h5')

# layer = model.get_layer(index=0)
# weights = layer.get_weights()
# print(weights)

# x = loadmat('Salinas_corrected.mat')
# data = x['salinas_corrected']
# print(data.shape)
# data = data[0:,0:,4:]
# print(data.shape)

# gx = loadmat('Salinas_gt.mat')
# gt = gx['salinas_gt']
# print(gt.shape)

