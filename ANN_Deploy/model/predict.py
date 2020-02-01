import numpy as np
from keras.models import load_model
classifier = load_model('model.h5')
X = np.array([[5.7, 2.8, 4.1, 1.3]])
predict = classifier.predict_classes(X)[0]
print(predict)
