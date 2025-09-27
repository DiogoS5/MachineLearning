import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import matplotlib.pyplot as plt

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")


# Split data into train and test sets
x_train, X_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(X_test)


# --- Kernel Ridge Regression (KRR) with RBF kernel ---
param_grid_krr = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'gamma': [0.01, 0.1, 1.0, 10.0]
}
grid_krr = GridSearchCV(
    KernelRidge(kernel='rbf'),
    param_grid_krr,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_krr.fit(x_train_scaled, y_train)
print(f"Best KRR parameters: {grid_krr.best_params_}")
best_krr = grid_krr.best_estimator_
y_pred_krr = best_krr.predict(X_test_scaled)

# Cross-validation scores for KRR
cv_scores_krr_r2 = cross_val_score(best_krr, x_train_scaled, y_train, cv=5, scoring='r2')
cv_scores_krr_mae = cross_val_score(best_krr, x_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_scores_krr_mse = cross_val_score(best_krr, x_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"KRR CV R2: {cv_scores_krr_r2.mean():.4f}")
print(f"KRR CV MAE: {-cv_scores_krr_mae.mean():.4f}")
print(f"KRR CV MSE: {-cv_scores_krr_mse.mean():.4f}")

# Test metrics for KRR
r2_krr = r2_score(y_test, y_pred_krr)
mae_krr = mean_absolute_error(y_test, y_pred_krr)
mse_krr = mean_squared_error(y_test, y_pred_krr)
rmse_krr = np.sqrt(mse_krr)
print(f'KRR Test R2: {r2_krr:.4f}')
print(f'KRR Test MAE: {mae_krr:.4f}')
print(f'KRR Test MSE: {mse_krr:.4f}')
print(f'KRR Test RMSE: {rmse_krr:.4f}')

# --- Support Vector Regression (SVR) with RBF kernel ---
param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5, 1]
}
grid_svr = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid_svr,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_svr.fit(x_train_scaled, y_train)
print(f"Best SVR parameters: {grid_svr.best_params_}")
best_svr = grid_svr.best_estimator_
y_pred_svr = best_svr.predict(X_test_scaled)

# Cross-validation scores for SVR
cv_scores_svr_r2 = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='r2')
cv_scores_svr_mae = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_scores_svr_mse = cross_val_score(best_svr, x_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"SVR CV R2: {cv_scores_svr_r2.mean():.4f}")
print(f"SVR CV MAE: {-cv_scores_svr_mae.mean():.4f}")
print(f"SVR CV MSE: {-cv_scores_svr_mse.mean():.4f}")

# Test metrics for SVR
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
print(f'SVR Test R2: {r2_svr:.4f}')
print(f'SVR Test MAE: {mae_svr:.4f}')
print(f'SVR Test MSE: {mse_svr:.4f}')
print(f'SVR Test RMSE: {rmse_svr:.4f}')

# Visualize predictions vs true values for both models
plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_krr, color='blue', label='KRR Predicted vs True', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.title('KRR (RBF) Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svr, color='red', label='SVR Predicted vs True', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.title('SVR (RBF) Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()

