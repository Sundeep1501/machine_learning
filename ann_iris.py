from sklearn import datasets
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense

iris = datasets.load_iris()

x = iris.data
y = to_categorical(iris.target)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(input_dim = 4, units=12, activation="relu"))
classifier.add(Dense(units=3, activation="softmax"))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train,
               batch_size = 10,
               epochs = 150,
               validation_data = (X_test, y_test))

score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#classifier.save('model.h5')
