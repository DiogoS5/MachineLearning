import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

GAMMA = 0.1
N_COMPONENTS = 200
N_SPLITS = 5
RANDOM_STATE = 42

RIDGE_ALPHAS = np.logspace(-3, 3, 13)    # 1e-3 ... 1e3
LASSO_ALPHAS = np.logspace(-4, 1, 13)    # 1e-4 ... 1e1

def radius_basis_function_transform_fit(X_training_data, X_validation_fold, gamma, n_components):
    scaler = StandardScaler().fit(X_training_data)
    X_training_scaled = scaler.transform(X_training_data)
    X_validation_scaled = scaler.transform(X_validation_fold)

    model = RBFSampler(gamma = gamma, n_components = n_components, random_state = RANDOM_STATE).fit(X_training_scaled)
    
    X_train_radius_basis_function_features = model.transform(X_training_scaled)
    X_validation_radius_basis_function_features = model.transform(X_validation_scaled)
    
    return X_train_radius_basis_function_features, X_validation_radius_basis_function_features

def radius_basis_function_transform_full(X_training_data, gamma, n_components):
    scaler_full = StandardScaler().fit(X_training_data)
    X_data_scaled = scaler_full.transform(X_training_data)
    
    radius_basis_function_full = RBFSampler(gamma = gamma, n_components = n_components, random_state = RANDOM_STATE).fit(X_data_scaled)
    X_radius_basis_function_features = radius_basis_function_full.transform(X_data_scaled)
    
    return X_radius_basis_function_features, scaler_full, radius_basis_function_full

def cross_validation_mean_r2_for_alpha(model_name, alphas, X_training_data, Y_training_data, gamma = GAMMA, n_components = N_COMPONENTS, n_splits = N_SPLITS):
    cross_validation_curve = []
    
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = RANDOM_STATE)

    for alpha_index in alphas:
        fold_scores = []
        
        for training_index, validation_index in kfold.split(X_training_data):
            X_traning_data_set, X_validation_set = X_training_data[training_index], X_training_data[validation_index]
            Y_training_set, Y_validation_set = Y_training_data[training_index], Y_training_data[validation_index]

            radius_basis_function_features_training, radius_basis_function_features_validation = radius_basis_function_transform_fit(X_traning_data_set, X_validation_set, gamma, n_components)

            if model_name == "ridge":
                model = Ridge(alpha = alpha_index)
            elif model_name == "lasso":
                model = Lasso(alpha = alpha_index, max_iter = 20000)
            else:
                raise ValueError("Unknown model name")

            model.fit(radius_basis_function_features_training, Y_training_set)
            
            fold_scores.append(model.score(radius_basis_function_features_validation, Y_validation_set))

        cross_validation_curve.append(float(np.mean(fold_scores)))

    cross_validation_curve = np.array(cross_validation_curve, dtype = float)
    best_index = int(np.argmax(cross_validation_curve))
    best_alpha = float(alphas[best_index])
    best_cross_validation_mean = float(cross_validation_curve[best_index])
    
    return best_alpha, best_cross_validation_mean, cross_validation_curve

def main():
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")

    X_training_set, X_test_set, Y_training_set, Y_test_set = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)

    ridge_best_alpha, ridge_best_cv, ridge_curve = cross_validation_mean_r2_for_alpha(
        model_name = "ridge",
        alphas = RIDGE_ALPHAS,
        X_training_data = X_training_set,
        Y_training_data = Y_training_set,
        gamma = GAMMA,
        n_components = N_COMPONENTS,
        n_splits = N_SPLITS
    )

    lasso_best_alpha, lasso_best_cv, lasso_curve = cross_validation_mean_r2_for_alpha(
        model_name = "lasso",
        alphas = LASSO_ALPHAS,
        X_training_data = X_training_set,
        Y_training_data = Y_training_set,
        gamma = GAMMA,
        n_components = N_COMPONENTS,
        n_splits = N_SPLITS
    )

    features_training_full, scaler_full, rbf_full = radius_basis_function_transform_full(X_training_set, GAMMA, N_COMPONENTS)
    features_test_full = rbf_full.transform(scaler_full.transform(X_test_set))

    ridge_final = Ridge(alpha = ridge_best_alpha).fit(features_training_full, Y_training_set)
    lasso_final = Lasso(alpha = lasso_best_alpha, max_iter = 20000).fit(features_training_full, Y_training_set)

    ridge_test_r2_score = r2_score(Y_test_set, ridge_final.predict(features_test_full))
    lasso_test_r2_sore = r2_score(Y_test_set, lasso_final.predict(features_test_full))

    print("=== RBF feature comparison (γ=%.3g, m=%d) ===" % (GAMMA, N_COMPONENTS))
    print(f"Ridge   — best alpha: {ridge_best_alpha:.4g}, CV R²: {ridge_best_cv:.4f}, TEST R²: {ridge_test_r2_score:.4f}")
    print(f"Lasso   — best alpha: {lasso_best_alpha:.4g}, CV R²: {lasso_best_cv:.4f}, TEST R²: {lasso_test_r2_sore:.4f}")

    plt.figure(figsize = (12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogx(RIDGE_ALPHAS, ridge_curve, marker='o')
    plt.axvline(ridge_best_alpha, linestyle='--')
    plt.title("Ridge (RBF) — mean R² vs alpha")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean CV R²")

    plt.subplot(1, 2, 2)
    plt.semilogx(LASSO_ALPHAS, lasso_curve, marker='o')
    plt.axvline(lasso_best_alpha, linestyle='--')
    plt.title("Lasso (RBF) — mean R² vs alpha")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("mean CV R²")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
