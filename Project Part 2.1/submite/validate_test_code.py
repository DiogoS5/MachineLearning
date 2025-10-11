
import numpy as np
import pandas as pd
from mymodel import predict
X_df = pd.read_pickle("Xtrain1.pkl")
y_test = np.load("Ytrain1.npy")

X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)
assert X.shape[1] == 132, f"Expected 132 features, got {X.shape[1]}"
# Make the predictions
y_pred = predict(X)
#validate the size of y_pred
if y_pred.shape != y_test.shape:
    raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_test.shape}")  
print("Prediction format is valid.")