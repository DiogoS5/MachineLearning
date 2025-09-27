
import numpy as np
from mymodel import predict
#Generate dummy random data X of size (300,6) and y of size (300,)
X_test = np.random.rand(300, 6)
y_test = np.random.rand(300)
# Make the predictions
y_pred = predict(X_test)
#validate the size of y_pred
if y_pred.shape != y_test.shape:
    raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_test.shape}")  
print("Prediction format is valid.")