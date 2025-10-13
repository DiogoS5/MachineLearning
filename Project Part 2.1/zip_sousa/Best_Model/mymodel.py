# mymodel.py
import sys
import numpy as np
import joblib
from model_utils import Sanitize, PoseNorm1, SymmetryFeatures, MirrorLeftToRightBody

_main = sys.modules.get('__main__')
if _main is not None:
    setattr(_main, 'Sanitize', Sanitize)
    setattr(_main, 'PoseNorm1', PoseNorm1)
    setattr(_main, 'MirrorLeftToRightBody', MirrorLeftToRightBody)
    setattr(_main, 'SymmetryFeatures', SymmetryFeatures)

_MODEL_PATH = "svc_rbf_final.pkl"
_EXPECTED_DIMS = 132

def predict(Xtest):
    Xtest = np.asarray(Xtest)
    
    if Xtest.ndim != 2 or Xtest.shape[1] != _EXPECTED_DIMS:
        raise ValueError(f"Xtest must have shape (N, {_EXPECTED_DIMS}), got {Xtest.shape}.")
    
    Xtest = Xtest.astype(float, copy=False)

    model = joblib.load(_MODEL_PATH)
    
    y_pred = model.predict(Xtest)
    
    return np.asarray(y_pred).reshape(-1,)
