#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

X_PATH = "Xtrain2.pkl"
Y_PATH = "Ytrain2.npy"
PID_COL = "Patient_Id"
SEQ_COL = "Skeleton_Sequence"
SEED = 42
N_SPLITS = 2
NK = 33  # number of keypoints

class SeqStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    From sequences shaped (T, NK, 2), compute:
      - per-KP mean (NKx2)
      - per-KP std  (NKx2)
      - per-KP avg speed (mean ||Δ||)
      - per-KP total travel (sum ||Δ||)
    Output per sample length = 4*NK + 4*NK = 2*NK*2 + NK + NK = 462 (for NK=33).
    """
    def __init__(self, seq_col=SEQ_COL, n_keypoints=NK, n_dims=2):
        self.seq_col = seq_col
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims
        self.feature_dim_ = n_keypoints * n_dims * 2 + n_keypoints + n_keypoints

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

            diffs  = np.diff(arr, axis=0)           # (T-1, NK, 2)
            speeds = np.linalg.norm(diffs, axis=2)  # (T-1, NK)
            if speeds.size > 0:
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
    def __init__(self, seq_col="Skeleton_Sequence"):
        self.seq_col = seq_col
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        feats = []
        for seq in X[self.seq_col].values:
            a = np.asarray(seq, float).reshape(-1, NK, 2)
            if a.shape[0] < 3:  # guard for very short sequences
                # build zeros of the right size: 4 stats for speed + 4 stats for accel, each NK long
                feats.append(np.zeros(8 * NK, dtype=float))
                continue
            v = np.diff(a, axis=0)       # (T-1, NK, 2)
            acc = np.diff(v, axis=0)     # (T-2, NK, 2)
            spd = np.linalg.norm(v, axis=2)      # (T-1, NK)
            accn = np.linalg.norm(acc, axis=2)   # (T-2, NK)
            stat = lambda z: np.concatenate([
                np.nanmean(z, 0), np.nanstd(z, 0),
                np.nanpercentile(z, 25, 0), np.nanpercentile(z, 75, 0)
            ], axis=None)
            feats.append(np.concatenate([stat(spd), stat(accn)], axis=0))
        return np.asarray(feats)

class PoseNorm1(BaseEstimator, TransformerMixin):
    """
    Normalize the first 4*NK columns (means NK*2 + stds NK*2) using:
      - centering by torso centroid (shoulders+hips)
      - anisotropic scaling: x/shoulder_distance, y/head-to-mid-hip distance
    Leaves remaining columns (e.g., speeds, travels, temporal extras) untouched if keep_rest=True.
    """
    def __init__(self, n_keypoints=NK, eps=1e-9, keep_rest=True):
        self.n_keypoints = n_keypoints
        self.eps = eps
        self.keep_rest = keep_rest

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        nk = self.n_keypoints

        coord_cols = 4 * nk  # NK*2 means + NK*2 stds
        if X.shape[1] < coord_cols:
            raise ValueError(f"PoseNorm1 expects at least {coord_cols} columns (got {X.shape[1]}).")

        mean_coords = X[:, :2 * nk].reshape(n_samples, nk, 2)
        std_coords  = X[:, 2 * nk:coord_cols].reshape(n_samples, nk, 2)
        rest = X[:, coord_cols:]  # speeds, travels, temporal extras, etc.

        shoulders = [11, 12]
        hips = [23, 24]
        head_idx = 0
        torso_indices = shoulders + hips  # [11, 12, 23, 24]

        torso_centroid = np.nanmean(mean_coords[:, torso_indices], axis=1)  # (n_samples, 2)

        shoulder_distance = np.linalg.norm(
            mean_coords[:, shoulders[0]] - mean_coords[:, shoulders[1]], axis=1
        )  # (n_samples,)

        mid_hip = np.nanmean(mean_coords[:, hips], axis=1)  # (n_samples, 2)
        vertical_distance = np.linalg.norm(
            mean_coords[:, head_idx] - mid_hip, axis=1
        )  # (n_samples,)

        # Avoid divide-by-zero
        shoulder_distance = np.where(shoulder_distance == 0, 1.0, shoulder_distance)
        vertical_distance = np.where(vertical_distance == 0, 1.0, vertical_distance)

        # Center at torso centroid
        mean_centered = mean_coords - torso_centroid[:, None, :]

        # Anisotropic scaling
        mean_normalized = np.empty_like(mean_centered)
        mean_normalized[:, :, 0] = mean_centered[:, :, 0] / shoulder_distance[:, None]
        mean_normalized[:, :, 1] = mean_centered[:, :, 1] / vertical_distance[:, None]

        std_normalized = np.empty_like(std_coords)
        std_normalized[:, :, 0] = std_coords[:, :, 0] / shoulder_distance[:, None]
        std_normalized[:, :, 1] = std_coords[:, :, 1] / vertical_distance[:, None]

        coord_out = np.concatenate(
            [mean_normalized.reshape(n_samples, -1),
             std_normalized.reshape(n_samples, -1)],
            axis=1
        )

        if self.keep_rest and rest.size:
            return np.concatenate([coord_out, rest], axis=1)
        return coord_out



X_df = pd.read_pickle(X_PATH)
y_patients = np.load(Y_PATH).astype(int)

def build_patient_mapping_sorted(patients_series, y_patients):
    uniq_sorted = np.sort(patients_series.unique())
    return dict(zip(uniq_sorted, y_patients.astype(int)))

patients_series = X_df[PID_COL]
patient_to_label = build_patient_mapping_sorted(patients_series, y_patients)
y = patients_series.map(patient_to_label).to_numpy().astype(int)
patients = patients_series.to_numpy()

print("unique(y):", np.unique(y, return_counts=True))

counts = patients_series.value_counts()
left_pids = [pid for pid in counts.index if patient_to_label[pid] == 0]
right_pids = [pid for pid in counts.index if patient_to_label[pid] == 1]

if len(left_pids) < 2 or len(right_pids) < 2:
    print("[WARN] Very few patients per class; consider smaller N_SPLITS.")

test_patients = np.array([left_pids[0], right_pids[0]])
train_patients = np.setdiff1d(patients_series.unique(), test_patients)

train_mask = np.isin(patients, train_patients)
test_mask  = np.isin(patients,  test_patients)

X_train_df, y_train, g_train = X_df.loc[train_mask], y[train_mask], patients[train_mask]
X_test_df,  y_test,  g_test  = X_df.loc[test_mask],  y[test_mask],  patients[test_mask]

print("Train y distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  y distribution:", dict(zip(*np.unique(y_test,  return_counts=True))))

sgkf = StratifiedGroupKFold(n_splits=min(N_SPLITS, len(np.unique(g_train))),shuffle=True,random_state=SEED)

feature_union = FeatureUnion([
    ("stats", SeqStatsFeaturizer(seq_col=SEQ_COL, n_keypoints=NK, n_dims=2)),
    ("temp",  TemporalExtras(seq_col=SEQ_COL)),
])

base_steps = [
    ("feats", feature_union),
    ("imputer", SimpleImputer(strategy="median")),
    ("pose_norm1", PoseNorm1(n_keypoints=NK, keep_rest=True)),
]

pipelines = {
    "rf": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]),
    "svc": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", SVC(class_weight="balanced", probability=False, random_state=SEED)),
    ]),
    "mlp": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                              alpha=1e-4, early_stopping=True, n_iter_no_change=20,
                              random_state=SEED)),
    ]),
    "knn": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ]),
    "et": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]),
    "logreg": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=2000, random_state=SEED)),
    ]),
    "gb": Pipeline(base_steps + [
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ]),
}

# -----------------------------
# Param grids
# -----------------------------
param_grids = {
    "rf": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
    },
    "svc": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.25, 0.5, 1, 2],
        "clf__gamma": ["scale", 0.1, 0.05],
    },
    "mlp": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3],
    },
    "knn": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "et": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },
    "logreg": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__C": [0.25, 0.5, 1, 2],
    },
    "gb": {
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.8, 1.0],
    },
}

# -----------------------------
# Train & evaluate
# -----------------------------
scores = {}
best_model = None
best_name = None
best_bal_acc = -np.inf

for name, pipe in pipelines.items():
    print(f"\nTraining {name} (StratifiedGroupKFold={sgkf.get_n_splits()}, scoring='balanced_accuracy')...")

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="balanced_accuracy",
        cv=sgkf,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )

    grid.fit(X_train_df, y_train, groups=g_train)

    y_pred = grid.predict(X_test_df)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"{name} — Test Balanced Accuracy: {bal_acc:.4f}")
    print("Best params:", grid.best_params_)

    scores[name] = bal_acc
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_model = grid.best_estimator_
        best_name = name

print(f"\nBest model: {best_name} | Test Balanced Accuracy: {best_bal_acc:.4f}")
print("Per-model Balanced Accuracy:", {k: f"{v:.4f}" for k, v in scores.items()})
