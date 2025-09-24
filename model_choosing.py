import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def fit_linear_model(training_data_X, training_data_Y, kfold):
    scores = []
    
    for traning, validation in kfold.split(training_data_X):
        model = LinearRegression().fit(training_data_X[traning], training_data_Y[traning])
        scores.append(model.score(training_data_X[validation], training_data_Y[validation])) #see if this score is r2
    
    final_model = LinearRegression().fit(training_data_X, training_data_Y)
    
    return ("linear", final_model), float(np.mean(scores))

def fit_polynomial_model(training_data_X, training_data_Y, kfold, polynominal_degrees = (2, 3, 4, 5, 6)):
    best_cross_validation_score, best_polynminal_degres, best_linear_model, best_polynominal_model = -np.inf, None, None, None
    
    for degrees in polynominal_degrees:
        polynominal = PolynomialFeatures(degree = degrees, include_bias = False)
        X_polynominal_matrix = polynominal.fit_transform(training_data_X)

        k_fold_validation_scores = []
        for training, validation in kfold.split(X_polynominal_matrix):
            model = LinearRegression().fit(X_polynominal_matrix[training], training_data_Y[training])
            k_fold_validation_scores.append(model.score(X_polynominal_matrix[validation], training_data_Y[validation])) #see if this score is r2
            
        mean_cross_validation_score = float(np.mean(k_fold_validation_scores))

        if mean_cross_validation_score > best_cross_validation_score:
            best_cross_validation_score, best_polynminal_degres = mean_cross_validation_score, degrees
            best_linear_model = LinearRegression().fit(X_polynominal_matrix, training_data_Y)
            best_polynominal_model = polynominal
            
    return ("poly", best_linear_model, best_polynominal_model, best_polynminal_degres), best_cross_validation_score

def fit_radius_basis_function_model(training_data_X, training_data_Y, kfold, gammas=np.logspace(-3, 1, 7, 9), components = (200, 500, 700, 1000)):
    best_cross_validation_score = -np.inf
    best_model = None
    for gamma in gammas:
        for m in components:
            
            fold_scores = []
            
            for trainning, validation in kfold.split(training_data_X):
                scaler = StandardScaler()
                X_training_scaled_data = scaler.fit_transform(training_data_X[trainning])
                X_validation_scaled_data = scaler.transform(training_data_X[validation])

                rbf = RBFSampler(gamma = gamma, n_components = m, random_state = 42)
                X_training_random_data = rbf.fit_transform(X_training_scaled_data)
                X_validation_random_data = rbf.transform(X_validation_scaled_data)

                linear_regression_model = LinearRegression().fit(X_training_random_data, training_data_Y[trainning])
                fold_scores.append(linear_regression_model.score(X_validation_random_data, training_data_Y[validation]))

            mean_cross_validation_score = float(np.mean(fold_scores))
            
            if mean_cross_validation_score > best_cross_validation_score:
                scaler_full = StandardScaler().fit(training_data_X)
                X_scaled_data = scaler_full.transform(training_data_X)
                
                rbf_full = RBFSampler(gamma=gamma, n_components=m, random_state=42).fit(X_scaled_data)
                X_random_data = rbf_full.transform(X_scaled_data)
                
                linear_model_full = LinearRegression().fit(X_random_data, training_data_Y)
                
                best_cross_validation_score = mean_cross_validation_score
                
                best_model = ("rbf", linear_model_full, scaler_full, rbf_full, gamma, m)
            
    return best_model, best_cross_validation_score

def predict(model_info, X_data):
    kind = model_info[0]
    
    if kind == "linear":
        _, lin = model_info
        
        return lin.predict(X_data)
    elif kind == "poly":
        _, lin, poly, _deg = model_info
        
        return lin.predict(poly.transform(X_data))
    elif kind == "rbf":
        _, lin, scaler, rbf, _g, _m = model_info
        X_scaled_date = scaler.transform(X_data)
        X_random_data = rbf.transform(X_scaled_date)
        
        return lin.predict(X_random_data)
    else:
        raise ValueError("Unknown model kind")

def main():
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")

    X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X_train, Y_train, test_size = 0.2, shuffle = True, random_state = 42)
    k_fold = KFold(n_splits = 10, shuffle = True, random_state = 42)

    linear_model_info, linear_model_cross_validation_score = fit_linear_model(X_train_split, Y_train_split, k_fold)
    polynominal_model_info, polynominal_model_cross_validation_model = fit_polynomial_model(X_train_split, Y_train_split, k_fold)
    radius_bassis_function_model_info,  radius_bassis_funcion_cross_validation_score = fit_radius_basis_function_model(X_train_split, Y_train_split, k_fold)

    print(f"Linear Model — Cross Validation R2: {linear_model_cross_validation_score:.4f}")
    print(f"Polynomial Model (best degree = {polynominal_model_info[3]}) — Cross Validation R2: {polynominal_model_cross_validation_model:.4f}")
    print(f"Radius Basis Function Model (gamma = {radius_bassis_function_model_info[4]:.4f}, n_componenets = {radius_bassis_function_model_info[5]}) — Cross Validation R2: {radius_bassis_funcion_cross_validation_score:.4f}")

    linear_model_r2_score  = r2_score(Y_test_split, predict(linear_model_info,  X_test_split))
    polynominal_model_r2_score = r2_score(Y_test_split, predict(polynominal_model_info, X_test_split))
    radius_bassis_function_r2_score  = r2_score(Y_test_split, predict(radius_bassis_function_model_info,  X_test_split))

    print("\nTEST R2:")
    print(f"  Linear Model:{linear_model_r2_score:.4f}")
    print(f"  Polynomial Model (degree = {polynominal_model_info[3]}): {polynominal_model_r2_score:.4f}")
    print(f"  Radius Basis Function Model (gamma = {radius_bassis_function_model_info[4]:.4f}, n_componenets = {radius_bassis_function_model_info[5]}): {radius_bassis_function_r2_score:.4f}")

    choices = [("Linear Model", linear_model_info, linear_model_r2_score),
               (f"Polynominal Model (degree = {polynominal_model_info[3]})", polynominal_model_info, polynominal_model_r2_score),
               (f"RBF Model (gamma = {radius_bassis_function_model_info[4]:.4f}, n_componenets = {radius_bassis_function_model_info[5]})", radius_bassis_function_model_info, radius_bassis_function_r2_score)]
    
    best_name, best_info, best_test = max(choices, key = lambda t: t[2])

    print(f"Best TEST R2: {best_name} (TEST R2={best_test:.4f})")

    y_pred = predict(best_info, X_test_split)
    
    plt.plot(np.arange(len(Y_test_split)), Y_test_split, label="True")
    plt.plot(np.arange(len(Y_test_split)), y_pred, label=f"{best_name}")
    plt.title(f"{best_name} — Test R2={best_test:.3f}")
    plt.xlabel("Sample index")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
