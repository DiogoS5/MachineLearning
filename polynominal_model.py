import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features = poly.fit_transform(X_train)

poly_reg_model = LinearRegression()

poly_reg_model.fit(poly_features, Y_train)

y_predicted = poly_reg_model.predict(poly_features)

plt.plot(np.arange(0, len(X_train)), Y_train)
plt.plot(np.arange(0, len(X_train)), y_predicted)
plt.show()

 