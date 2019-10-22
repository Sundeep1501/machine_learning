import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 2, 3]).reshape((-1, 1))
Y = np.array([1, 2, 3, 6])

model = LinearRegression().fit(X, Y)

print("Slope", model.coef_)
print("Intercept", model.intercept_)

X = np.array([1, 2, 2, 3,4,5,6,7,8,9]).reshape((-1, 1))

print("When x is 4, y is ", model.predict(X),sep='\n')

