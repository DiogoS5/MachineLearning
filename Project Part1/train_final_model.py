import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

#Load data
x_train = np.load("X_train.npy")
y_train = np.load("Y_train.npy")

# Outlier removal (IQR)
def remove_outliers(Xt, yt):
    Q1 = np.percentile(Xt, 25, axis=0)
    Q3 = np.percentile(Xt, 75, axis=0)
    IQR = Q3 - Q1
    mask_X = np.all((Xt >= (Q1 - 1.5 * IQR)) & (Xt <= (Q3 + 1.5 * IQR)), axis=1)

    y_Q1 = np.percentile(yt, 25)
    y_Q3 = np.percentile(yt, 75)
    y_IQR = y_Q3 - y_Q1
    mask_y = (yt >= (y_Q1 - 1.5 * y_IQR)) & (yt <= (y_Q1 + 1.5 * y_IQR))

    mask = mask_X & mask_y
    return Xt[mask], yt[mask]

x_train, y_train = remove_outliers(x_train, y_train)

#Build pipeline: scaler -> SVR with your fixed params
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01))
])

# Cross-validation on the cleaned training set
cv_r2  = cross_val_score(pipe, x_train, y_train, cv=5, scoring="r2").mean()
cv_mae = -cross_val_score(pipe, x_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
cv_mse = -cross_val_score(pipe, x_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()

print(f"CV R²:  {cv_r2:.4f}")
print(f"CV MAE: {cv_mae:.4f}")
print(f"CV MSE: {cv_mse:.4f}")

#Fit teh model
pipe.fit(x_train, y_train)

joblib.dump(pipe, "svr_model_pickle.pkl")
