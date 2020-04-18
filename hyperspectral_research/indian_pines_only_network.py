import numpy as np
import keras
from keras.models import load_model
import os.path


X_train = np.load('X_train'+'.npy')
Y_train = np.load('Y_train'+'.npy')
X_test = np.load('X_test'+'.npy')
Y_test = np.load('Y_test'+'.npy')

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
                activation='relu'
                #kernel_initializer = keras.initializers.constant(weights),
                #trainable=False
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
model.save('indian_pines_11.h5')