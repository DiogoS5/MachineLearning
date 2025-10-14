#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings

# Silence known sklearn warnings about single-class folds (harmless)
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
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

# -----------------------------
# Config
# -----------------------------
X_PATH = "Xtrain2.pkl"
Y_PATH = "Ytrain2.npy"
PID_COL = "Patient_Id"
SEQ_COL = "Skeleton_Sequence"
SEED = 42
N_SPLITS = 5

# -----------------------------
# Helpers
# -----------------------------
def seq_to_features(seq) -> np.ndarray:
    """Turn a sequence (T, ...) into a fixed-size vector: [mean,std,min,max] over time."""
    arr = np.asarray(seq)
    if arr.ndim == 1:
        T = arr.shape[0]
        arr2 = arr.reshape(T, 1)
    else:
        T = arr.shape[0]
        arr2 = arr.reshape(T, -1)
    mu = np.nanmean(arr2, axis=0)
    sd = np.nanstd(arr2, axis=0)
    mn = np.nanmin(arr2, axis=0)
    mx = np.nanmax(arr2, axis=0)
    return np.concatenate([mu, sd, mn, mx], axis=0)

def build_patient_mapping_sorted(patients_series: pd.Series, y_patients: np.ndarray):
    uniq_sorted = np.sort(patients_series.unique())
    return dict(zip(uniq_sorted, y_patients.astype(int)))

def make_stratified_group_kfold(groups: np.ndarray, y: np.ndarray, n_splits: int, seed: int):
    """
    Create CV folds over 'groups' (patients) such that each fold has both classes 0 and 1.
    Returns a list of (train_idx, test_idx) splits usable directly in GridSearchCV(cv=...).
    """
    rng = np.random.RandomState(seed)
    uniq_pids = pd.unique(groups)
    pid_to_label = {pid: int(np.round(np.mean(y[groups == pid]))) for pid in uniq_pids}

    left_pids  = [pid for pid in uniq_pids if pid_to_label[pid] == 0]
    right_pids = [pid for pid in uniq_pids if pid_to_label[pid] == 1]

    max_splits = min(n_splits, len(left_pids), len(right_pids))
    n_splits = max_splits

    rng.shuffle(left_pids)
    rng.shuffle(right_pids)

    buckets = [[] for _ in range(n_splits)]
    for i, pid in enumerate(left_pids):
        buckets[i % n_splits].append(pid)
    for i, pid in enumerate(right_pids):
        buckets[i % n_splits].append(pid)

    splits = []
    for b in buckets:
        test_mask = np.isin(groups, b)
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        if np.unique(y[test_idx]).size < 2:
            continue
        splits.append((train_idx, test_idx))

    # Print info about the created folds
    print(f"\nStratified Group K-Fold")
    print(f"Requested splits: {n_splits} | Actual splits created: {len(splits)}")
    return splits

X_df: pd.DataFrame = pd.read_pickle(X_PATH)
y_patients: np.ndarray = np.load(Y_PATH).astype(int)

X = np.vstack([seq_to_features(s) for s in X_df[SEQ_COL].to_list()]).astype(float)

patients_series = X_df[PID_COL]
patient_to_label = build_patient_mapping_sorted(patients_series, y_patients)
y = patients_series.map(patient_to_label).to_numpy().astype(int)
patients = patients_series.to_numpy()

print("unique(y):", np.unique(y, return_counts=True))


counts = patients_series.value_counts()
left_pids  = [pid for pid in counts.index if patient_to_label[pid] == 0]
right_pids = [pid for pid in counts.index if patient_to_label[pid] == 1]

test_patients  = np.array([left_pids[0], right_pids[0]])
train_patients = np.setdiff1d(np.sort(patients_series.unique()), test_patients)

print("Train patients:", np.sort(train_patients))
print("Test  patients:", np.sort(test_patients))

train_mask = np.isin(patients, train_patients)
test_mask  = np.isin(patients,  test_patients)

X_train, y_train, g_train = X[train_mask], y[train_mask], patients[train_mask]
X_test,  y_test,  g_test  = X[test_mask],  y[test_mask],  patients[test_mask]

print("Train y distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  y distribution:", dict(zip(*np.unique(y_test,  return_counts=True))))
if np.unique(y_test).size < 2:
    raise RuntimeError("Test set is single-class. Pick different test patients.")

pipelines = {
    # Existing
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]),
    "svc": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=SEED)),
    ]),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            alpha=1e-4,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=SEED,
        )),
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ]),
    "et": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]),
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED
        )),
    ]),
    "linsvc": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
    ]),
    "gb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ]),
}

param_grids = {
    "rf": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_estimators": [300, 600, 900],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
    },
    "svc": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.25, 0.5, 1, 2, 4],
        "clf__gamma": ["scale", 0.1, 0.05, 0.02],
    },
    "mlp": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
    },
    "knn": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "et": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_estimators": [300, 600, 1000],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "logreg": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__C": [0.25, 0.5, 1, 2, 4],
        "clf__penalty": ["l2"],
    },
    "linsvc": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__C": [0.25, 0.5, 1, 2, 4],
        "clf__loss": ["squared_hinge"],
        "clf__max_iter": [3000],
    },
    "gb": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.8, 1.0],
    },
}

cv_splits = make_stratified_group_kfold(g_train, y_train, n_splits=N_SPLITS+1, seed=SEED)

scores = {}
best_model = None
best_name = None
best_test_bal_acc = -np.inf

for name, pipe in pipelines.items():
    print(f"\nTraining {name} (GridSearchCV, StratifiedGroupKFold={len(cv_splits)}, scoring='balanced_accuracy')...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="balanced_accuracy",
        cv=cv_splits,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    y_pred_test = grid.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    print(f"{name} — Test Balanced Accuracy: {bal_acc:.4f}")
    print("Best params:", grid.best_params_)
    scores[name] = bal_acc

    if bal_acc > best_test_bal_acc:
        best_test_bal_acc = bal_acc
        best_model = grid.best_estimator_
        best_name = name

print(f"\nBest model (by Balanced Accuracy): {best_name} | Test Balanced Accuracy: {best_test_bal_acc:.4f}")
print("Per-model Balanced Accuracy:", {k: f"{v:.4f}" for k, v in scores.items()})
