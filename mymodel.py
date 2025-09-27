import joblib

def predict(X_data):
    loaded_model = joblib.load("svr_model_pickle.pkl")
    
    y_pred = loaded_model.predict(X_data)
    
    return y_pred
