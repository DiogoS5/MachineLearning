import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=40
)

# --- Outlier Removal using IQR ---
def remove_outliers(X, y):
    # For each feature in X, remove rows where any feature is an outlier
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((X >= (Q1 - 1.5 * IQR)) & (X <= (Q3 + 1.5 * IQR)), axis=1)
    # Also remove outliers in y
    y_Q1 = np.percentile(y, 25)
    y_Q3 = np.percentile(y, 75)
    y_IQR = y_Q3 - y_Q1
    y_mask = (y >= (y_Q1 - 1.5 * y_IQR)) & (y <= (y_Q3 + 1.5 * y_IQR))
    final_mask = mask & y_mask
    return X[final_mask], y[final_mask]

x_train, y_train = remove_outliers(x_train, y_train)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Support Vector Regression (SVR) with RBF kernel ---
param_grid_svr = {
    'C': [100],
    'gamma': [0.1],
    'epsilon': [0.01]
}
grid_svr = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid_svr,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_svr.fit(x_train_scaled, y_train)
print(f"Best parameters: {grid_svr.best_params_}")
best_svr = grid_svr.best_estimator_
y_pred_svr = best_svr.predict(x_test_scaled)

# Cross-validation scores for SVR
cv_scores_svr_r2 = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='r2')
cv_scores_svr_mae = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_scores_svr_mse = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"CV R2: {cv_scores_svr_r2.mean():.4f}")
print(f"CV MAE: {-cv_scores_svr_mae.mean():.4f}")
print(f"CV MSE: {-cv_scores_svr_mse.mean():.4f}")

# Test metrics for SVR
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
print(f'Test R2: {r2_svr:.4f}')
print(f'Test MAE: {mae_svr:.4f}')
print(f'Test MSE: {mse_svr:.4f}')
print(f'Test RMSE: {rmse_svr:.4f}')

# Visualize predictions vs true values
plt.scatter(y_test, y_pred_svr, color='red', label='SVR Predicted vs True', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.title('SVR (RBF) Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()

