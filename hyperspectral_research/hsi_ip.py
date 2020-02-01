import numpy as np
import keras
from keras.utils import to_categorical

data = np.load("indianpinearray.npy")
gt = np.load("IPgt.npy")
print(data.shape)
print(gt.shape)

x = np.empty((145*145, 200))
y = np.array([])

index = 0

for i in range(145):
    for j in range(145):
        x[index] = np.array(data[i][j])
        y = np.append(y, gt[i][j])
        index = index+1

print(x.shape)
print(y.shape)

y = to_categorical(y)
print(y.shape)