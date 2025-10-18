import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from collections import defaultdict
from statistics import mode

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.exceptions import UndefinedMetricWarning

X_PATH = "Xtrain2.pkl"
Y_PATH = "Ytrain2.npy"
PID_COL = "Patient_Id"
SEQ_COL = "Skeleton_Sequence"
SEED = 42
N_SPLITS = 2
NK = 33
TEST_SIZE = 0.25
CACHE_DIR = None

SAVE_BEST = True
SAVE_DIR = Path("Best_Model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PKL = SAVE_DIR / "best_model_task2.pkl"

USE_STATS    = True
USE_TEMPORAL = True
USE_POSTURE  = True

class SeqStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    From sequences shaped (T, NK, 2), compute:
      - per-KP mean (NK*2)
      - per-KP std  (NK*2)
      - per-KP avg speed (NK)  [mean ||Δ||]
      - per-KP total travel (NK) [sum ||Δ||]
    => total = 6*NK
    """
    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK, n_dims=2):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SeqStatsFeaturizer expects a pandas DataFrame input.")
        n_samples = len(X)

        means = np.zeros((n_samples, self.n_keypoints, self.n_dims))
        stds = np.zeros((n_samples, self.n_keypoints, self.n_dims))
        avg_speeds = np.zeros((n_samples, self.n_keypoints))
        travels = np.zeros((n_samples, self.n_keypoints))

        for i, seq in enumerate(X[self.seq_col].values):
            arr = np.asarray(seq, dtype=float)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, self.n_keypoints, self.n_dims)

            means[i] = np.nanmean(arr, axis=0)
            stds[i]  = np.nanstd(arr, axis=0)

            diffs  = np.diff(arr, axis=0)  # (T-1, NK, 2)
            if diffs.size > 0:
                speeds = np.linalg.norm(diffs, axis=2)  # (T-1, NK)
                avg_speeds[i] = np.nanmean(speeds, axis=0)
                travels[i]    = np.nansum(speeds, axis=0)

        feats = np.concatenate(
            [
                means.reshape(n_samples, -1),   # NK*2
                stds.reshape(n_samples, -1),    # NK*2
                avg_speeds,                     # NK
                travels,                        # NK
            ],
            axis=1,
        )
        return feats


class TemporalExtras(BaseEstimator, TransformerMixin):
    """
    Speed/acceleration distribution stats per keypoint:
    For each of speed and accel: mean, std, p25, p75  (each NK long) => 4*NK per set => total 8*NK.
    """
    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for seq in X[self.seq_col].values:
            a = np.asarray(seq, float).reshape(-1, self.n_keypoints, 2)
            # Need at least 3 frames to compute accel
            if a.shape[0] < 3:
                feats.append(np.zeros(8 * self.n_keypoints, dtype=float))
                continue

            v = np.diff(a, axis=0)       # (T-1, NK, 2)
            acc = np.diff(v, axis=0)     # (T-2, NK, 2)
            spd = np.linalg.norm(v, axis=2)      # (T-1, NK)
            accn = np.linalg.norm(acc, axis=2)   # (T-2, NK)

            def stat(z):  # returns 4*NK
                return np.concatenate([
                    np.nanmean(z, axis=0),                     # NK
                    np.nanstd(z, axis=0),                      # NK
                    np.nanpercentile(z, 25, axis=0),          # NK
                    np.nanpercentile(z, 75, axis=0),          # NK
                ], axis=None)

            # total 8*NK
            feats.append(np.concatenate([stat(spd), stat(accn)], axis=0))

        return np.asarray(feats)


class PostureFeaturizer(BaseEstimator, TransformerMixin):
    """
    Adds:
      - ranges: per-KP motion range across time: ||max - min|| (NK)
      - right_vs_left: scalar = sum(std on right arm) - sum(std on left arm)
      - posture_angle: scalar = atan2(torso_vec.x, torso_vec.y)   # angle vs vertical (y)
    """
    LEFT_ARM  = [14,16,18,20,22]  # indices for left arm chain
    RIGHT_ARM = [13,15,17,19,21]  # indices for right arm chain
    SHOULDERS = [11,12]
    HIPS      = [23,24]

    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples = len(X)
        ranges = np.zeros((n_samples, self.n_keypoints))
        right_vs_left = np.zeros((n_samples, 1))
        posture_angle = np.zeros((n_samples, 1))

        for i, seq in enumerate(X[self.seq_col].values):
            arr = np.asarray(seq, dtype=float)
            if arr.size == 0:
                continue
            a = arr.reshape(-1, self.n_keypoints, 2)

            # Ranges per keypoint
            a_max = np.nanmax(a, axis=0)
            a_min = np.nanmin(a, axis=0)
            ranges[i] = np.linalg.norm(a_max - a_min, axis=1)

            # Means/stds for posture & side activity
            means = np.nanmean(a, axis=0)     # (NK,2)
            stds  = np.nanstd(a, axis=0)      # (NK,2)

            # Right-vs-left movement intensity (sum std norms on arms)
            left_std  = np.linalg.norm(stds[self.LEFT_ARM], axis=1)   # (len arm)
            right_std = np.linalg.norm(stds[self.RIGHT_ARM], axis=1)
            right_vs_left[i, 0] = np.nansum(right_std) - np.nansum(left_std)

            # Posture angle (torso vector from shoulders to hips centroid, angle wrt +y axis)
            torso_vec = np.nanmean(means[self.HIPS], axis=0) - np.nanmean(means[self.SHOULDERS], axis=0)
            posture_angle[i, 0] = np.arctan2(torso_vec[0], torso_vec[1])

        return np.concatenate([ranges, right_vs_left, posture_angle], axis=1)



X_df = pd.read_pickle(X_PATH)
y_patients = np.load(Y_PATH).astype(int)

def build_patient_mapping_sorted(p_series, y_arr):
    uniq_sorted = np.sort(p_series.unique())
    return dict(zip(uniq_sorted, y_arr.astype(int)))

patients_series = X_df[PID_COL]
patient_to_label = build_patient_mapping_sorted(patients_series, y_patients)
y = patients_series.map(patient_to_label).to_numpy().astype(int)
groups = patients_series.to_numpy()

print("unique(y):", np.unique(y, return_counts=True))

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train_df, y_train, g_train = X_df.iloc[train_idx], y[train_idx], groups[train_idx]
X_test_df,  y_test,  g_test  = X_df.iloc[test_idx],  y[test_idx],  groups[test_idx]

print("Train y distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  y distribution:", dict(zip(*np.unique(y_test,  return_counts=True))))

sgkf = StratifiedGroupKFold(n_splits=min(N_SPLITS, np.unique(g_train).size),shuffle=True, random_state=SEED)

feats_list = []
if USE_STATS:
    feats_list.append(("stats", SeqStatsFeaturizer(seq_col=SEQ_COL, n_keypoints=NK, n_dims=2)))
if USE_TEMPORAL:
    feats_list.append(("temp", TemporalExtras(seq_col=SEQ_COL, n_keypoints=NK)))
if USE_POSTURE:
    feats_list.append(("posture", PostureFeaturizer(seq_col=SEQ_COL, n_keypoints=NK)))

if not feats_list:
    raise RuntimeError("Enable at least one of USE_STATS/USE_TEMPORAL/USE_POSTURE.")

feature_union = FeatureUnion(feats_list)

base_steps = [
    ("feats", feature_union),
    ("imputer", SimpleImputer(strategy="median")),
]

def make_pipe(clf):
    return Pipeline(base_steps + [("scaler", StandardScaler()), ("clf", clf)], memory=CACHE_DIR)

pipelines = {
    "rf":     make_pipe(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    "svc":    make_pipe(SVC(class_weight="balanced", probability=False, random_state=SEED)),
    "mlp":    make_pipe(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, alpha=1e-4,
                                      early_stopping=True, n_iter_no_change=20, random_state=SEED)),
    "knn":    make_pipe(KNeighborsClassifier()),
    "logreg": make_pipe(LogisticRegression(class_weight="balanced", solver="liblinear",
                                           max_iter=2000, random_state=SEED)),
}

param_grids = {
    "rf": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators":[300,600],
        "clf__max_depth":[None,12,18],
        "clf__min_samples_split":[2,4],
    },
    "svc": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__kernel":["linear","rbf"],
        "clf__C":[0.25,0.5,1,2],
        "clf__gamma":["scale",0.1,0.05],
    },
    "mlp": {
        "scaler":[StandardScaler(), RobustScaler()],
        "clf__hidden_layer_sizes":[(128,64),(256,128)],
        "clf__alpha":[1e-4,1e-3],
    },
    "knn": {
        "scaler":[StandardScaler(), RobustScaler()],
        "clf__n_neighbors":[3,5,7],
        "clf__weights":["uniform","distance"],
        "clf__p":[1,2],
    },
    "logreg": {
        "scaler":[StandardScaler(), RobustScaler()],
        "clf__C":[0.25,0.5,1,2],
    },
}

def aggregate_patient_preds(pids, row_preds):
    """Majority vote per patient (ties -> round(mean))."""
    bucket = defaultdict(list)
    for pid, p in zip(pids, row_preds):
        bucket[int(pid)].append(int(p))
    sorted_pids = np.array(sorted(bucket.keys()), dtype=int)
    y_pred_patient = []
    for pid in sorted_pids:
        votes = bucket[pid]
        try:
            maj = mode(votes)
        except Exception:
            maj = int(np.round(np.mean(votes)))
        y_pred_patient.append(maj)
    return sorted_pids, np.array(y_pred_patient, dtype=int)

scores_row   = {}
scores_pat   = {}
best = {"name": None, "bal_acc_pat": -np.inf, "est": None, "grid": None}

test_pids = X_test_df[PID_COL].to_numpy()

for name, pipe in pipelines.items():
    print(f"\nTraining {name} (CV={sgkf.get_n_splits()} folds; scoring=balanced_accuracy)...")
    grid = GridSearchCV(
        pipe, param_grids[name],
        scoring="balanced_accuracy",
        cv=sgkf,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(X_train_df, y_train, groups=g_train)

    cv_mean = grid.cv_results_["mean_test_score"][grid.best_index_]
    cv_std  = grid.cv_results_["std_test_score"][grid.best_index_]
    print(f"{name} — CV balanced acc: {cv_mean:.4f} ± {cv_std:.4f}")
    print("Best params:", grid.best_params_)

    y_pred_rows = grid.predict(X_test_df)
    print(f"{name} — y_pred (row-level, len={len(y_pred_rows)}):")
    print(y_pred_rows)

    bal_acc_row = balanced_accuracy_score(y_test, y_pred_rows)
    scores_row[name] = bal_acc_row
    print(f"{name} — TEST balanced acc (row-level): {bal_acc_row:.4f}")

    pids_sorted, y_pred_pat = aggregate_patient_preds(test_pids, y_pred_rows)
    y_true_pat = np.array([patient_to_label[pid] for pid in pids_sorted], dtype=int)

    bal_acc_pat = balanced_accuracy_score(y_true_pat, y_pred_pat)
    scores_pat[name] = bal_acc_pat
    print(f"{name} — TEST balanced acc (patient-level): {bal_acc_pat:.4f}")

    print("\nPatient-level predictions:")
    print("patient_ids:     ", pids_sorted)
    print("y_true_patient:  ", y_true_pat)
    print("y_pred_patient:  ", y_pred_pat)

    print("\nPatient-level Confusion matrix (rows=true, cols=pred):\n",confusion_matrix(y_true_pat, y_pred_pat))

    if bal_acc_pat > best["bal_acc_pat"]:
        best.update({"name": name, "bal_acc_pat": bal_acc_pat, "est": grid.best_estimator_, "grid": grid})

print("\nPer-model TEST balanced accuracy (row-level):",{k: f"{v:.4f}" for k, v in scores_row.items()})
print("Per-model TEST balanced accuracy (patient-level):",{k: f"{v:.4f}" for k, v in scores_pat.items()})
print(f"Best model by patient-level metric: {best['name']} | TEST balanced acc (patient): {best['bal_acc_pat']:.4f}")

best_est = best["est"]
y_hat_rows = best_est.predict(X_test_df)
pids_sorted, y_hat_pat = aggregate_patient_preds(test_pids, y_hat_rows)
y_true_pat = np.array([patient_to_label[pid] for pid in pids_sorted], dtype=int)
print("\n[Best model] Patient-level confusion:\n",confusion_matrix(y_true_pat, y_hat_pat))

from pathlib import Path
import joblib

SAVE_DIR = Path("Best_Model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PKL = SAVE_DIR / "best_model_task2.pkl"

final_pipe = pipelines[best["name"]]
final_pipe.set_params(**best["grid"].best_params_)

print("\nFitting BEST pipeline on ALL data …")
final_pipe.fit(X_df, y)

joblib.dump(final_pipe, MODEL_PKL)
print(f"Saved best pipeline to: {MODEL_PKL.resolve()}")
