# <<<<<<<<<<< PART 1 >>>>>>>>>>>
# DATE PREPROCESSING 

# IMPORT LIBRARIES
from spectral import *
import numpy as np

import keras
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import time

# load file name to save the results
filename = str(int(time.time()))+".txt"
f = open(filename, 'w')

# LOAD HYPERSPECTRAL AND GROUND TRUTH IMAGE
img = open_image('92AV3C.lan')
f.write(str(img))
f.write('\n\n')
gt = open_image('92AV3GT.GIS')
f.write(str(gt))

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

f.write("\nData samples available each class")
f.write("\n"+str(classes))

f.write("\nComplete Raw Data Shape")
f.write("\n"+str(X.shape))
f.write("\n"+str(Y.shape))


# SPLIT THE DATA INTO TRAINING SET AND TEST SET
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# FROM SINGLE GROUND TRUTH VALUES, CONVERT TO CATEGORICAL
Y_train_raw = Y_train
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

f.write("\nTraining and Test Data")

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

f.write("\n"+str(X_train.shape))
f.write("\n"+str(Y_train.shape))
f.write("\n"+str(X_test.shape))
f.write("\n"+str(Y_test.shape))

# <<<<<<<<<<<<<<<< PART 2 >>>>>>>>>>>>>>>>>
# Evaluating, Improving and Tuning the CNN


# Evaluting the CNN using k-fold cross validation to know the model accuracy and variance
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential
# from keras.layers import Conv1D
# from keras.layers import MaxPooling1D
# from keras.layers import Dense
# from keras.layers import Flatten

# def build_classifier():
#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=10, input_shape=(X_train.shape[1],1), activation='relu'))
#     model.add(MaxPooling1D(pool_size = 2))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(Y_train[0].shape[0], activation='softmax'))
#     model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#     return model

# classifier = KerasClassifier(build_fn = build_classifier, batch_size=32, epochs=100)
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
# f.write("\n\nAccuracy: " + str(accuracies.mean()))
# f.write("\nVariance: " + str(accuracies.std()))
# f.close()


# Improving the CNN
# Dropout regularization to reduce overfitting if needed
# Drop neurons in the layer after every iteration to reduce overfitting
# The model makes some neurons disappear so that neurons in the network are more independent.


# Tuning the CNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten

def build_classifier(optimizer):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10, input_shape=(X_train.shape[1],1), activation='relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(Y_train[0].shape[0], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size':[25, 32],
              'epochs':[100, 500],
              'optimizer':[keras.optimizers.Adam(), keras.optimizers.RMSprop()]}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train,Y_train_raw)

f.write("\n\nAccuracy: " + str(grid_search.best_params_))
f.write("\nVariance: " + str(grid_search.best_score_))

f.close()