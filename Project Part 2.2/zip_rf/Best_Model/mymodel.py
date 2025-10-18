from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sys
from features import SeqStatsFeaturizer, TemporalExtras, PostureFeaturizer

_this_main = sys.modules.get("__main__")
if _this_main is not None:
    setattr(_this_main, "SeqStatsFeaturizer", SeqStatsFeaturizer)
    setattr(_this_main, "TemporalExtras", TemporalExtras)
    setattr(_this_main, "PostureFeaturizer", PostureFeaturizer)

MODEL_PATH = Path(__file__).with_name("best_model_task2.pkl")
PID_COL = "Patient_Id"

def _aggregate_patient_preds(pids: np.ndarray, row_preds: np.ndarray) -> np.ndarray:
    pids = pids.astype(int)
    uniq = np.array(sorted(np.unique(pids)))
    out = []
    for pid in uniq:
        votes = row_preds[pids == pid].astype(int)
        ones, n = int(votes.sum()), int(votes.size)
        if ones * 2 > n:   maj = 1
        elif ones * 2 < n: maj = 0
        else:              maj = int(round(ones / n))
        out.append(maj)
    return np.asarray(out, dtype=int)

def predict(Xtest_df: pd.DataFrame):
    if not isinstance(Xtest_df, pd.DataFrame):
        raise TypeError("Xtest_df must be a pandas DataFrame.")
    if PID_COL not in Xtest_df.columns:
        raise ValueError(f"Xtest_df must contain column '{PID_COL}'.")

    pipe = joblib.load(MODEL_PATH)
    y_rows = pipe.predict(Xtest_df)
    pids = Xtest_df[PID_COL].to_numpy()
    y_pat = _aggregate_patient_preds(pids, y_rows)
    return y_pat
