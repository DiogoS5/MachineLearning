import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

X_PATH = "Xtrain2.pkl"
Y_PATH = "Ytrain2.npy"
PID_COL = "Patient_Id"
SEQ_COL = "Skeleton_Sequence"
SEED = 42
N_SPLITS = 2
NK = 33
EPS = 1e-9
TEST_SIZE = 0.25

CACHE_DIR = None

# Toggle ablations
USE_STATS    = True
USE_TEMPORAL = True
USE_POSTURE  = True


class SeqStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    From sequences shaped (T, NK, 2), compute:
      - per-KP mean (NKx2)
      - per-KP std  (NKx2)
      - per-KP avg speed (mean ||Δ||)
      - per-KP total travel (sum  ||Δ||)
    Output shape per sample: (NK*2 + NK*2 + NK + NK) = 6*NK*? Actually:
      means: NK*2
      stds:  NK*2
      avg_speeds: NK
      travels:    NK
      => total = 4*NK + 2*NK = 6*NK = 198 when NK=33, but means/stds are 2D each so 4*NK == 132; + 2*NK == 66; total 198.
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
    def __init__(self, seq_col=SEQ_COL):
        self.seq_col = seq_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for seq in X[self.seq_col].values:
            a = np.asarray(seq, float).reshape(-1, NK, 2)
            if a.shape[0] < 3:
                feats.append(np.zeros(8 * NK, dtype=float))
                continue
            v = np.diff(a, axis=0)       # (T-1, NK, 2)
            acc = np.diff(v, axis=0)     # (T-2, NK, 2)
            spd = np.linalg.norm(v, axis=2)      # (T-1, NK)
            accn = np.linalg.norm(acc, axis=2)   # (T-2, NK)

            def stat(z):
                return np.concatenate([
                    np.nanmean(z, 0), np.nanstd(z, 0),
                ], axis=None)

            feats.append(np.concatenate([stat(spd), stat(accn)], axis=0))
        return np.asarray(feats)


class PostureFeaturizer(BaseEstimator, TransformerMixin):
    """
    Adds:
      - ranges: per-KP motion range across time: ||max - min|| (NK)
      - right_vs_left: scalar = sum(std on right arm) - sum(std on left arm)
      - posture_angle: scalar = atan2(torso_vec.x, torso_vec.y)   # angle vs vertical (y)
    """
    LEFT_KPS  = [12,14,16,18,20,22,24,26,28,30,32]
    RIGHT_KPS = [11,13,15,17,19,21,23,25,27,29,31]
    SHOULDERS = [11,12]
    HIPS      = [23,24]

    LEFT_ARM  = [14,16,18,20,22]  # shoulder-elbow-wrist-hand chain approx
    RIGHT_ARM = [13,15,17,19,21]

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
            left_std  = np.linalg.norm(stds[self.LEFT_ARM], axis=1)   # (len arm), norm across x,y
            right_std = np.linalg.norm(stds[self.RIGHT_ARM], axis=1)
            right_vs_left[i, 0] = np.nansum(right_std) - np.nansum(left_std)

            # Posture angle (torso vector from shoulders to hips centroid, angle wrt vertical axis)
            torso_vec = np.nanmean(means[self.HIPS], axis=0) - np.nanmean(means[self.SHOULDERS], axis=0)
            # angle relative to +y axis: atan2(x, y)
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

sgkf = StratifiedGroupKFold(n_splits=min(N_SPLITS, np.unique(g_train).size), shuffle=True, random_state=SEED)

feature_union = FeatureUnion([
    ("stats",   SeqStatsFeaturizer(seq_col=SEQ_COL, n_keypoints=NK, n_dims=2)),
    ("temp",    TemporalExtras(seq_col=SEQ_COL)),
    ("posture", PostureFeaturizer(seq_col=SEQ_COL, n_keypoints=NK)),
])

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
    "et":     make_pipe(ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    "logreg": make_pipe(LogisticRegression(class_weight="balanced", solver="liblinear",
                                           max_iter=2000, random_state=SEED)),
    "gb":     make_pipe(GradientBoostingClassifier(random_state=SEED)),
}

param_grids = {
    "rf": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators":[300,600], 
        "clf__max_depth":[None,12,18], 
        "clf__min_samples_split":[2,4]
    },
    "svc": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__kernel":["linear","rbf"],
        "clf__C":[0.25,0.5,1,2], 
        "clf__gamma":["scale",0.1,0.05]
    },
    "mlp": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__hidden_layer_sizes":[(128,64),(256,128)], 
        "clf__alpha":[1e-4,1e-3]
    },
    "knn": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_neighbors":[3,5,7],
        "clf__weights":["uniform","distance"], 
        "clf__p":[1,2]
    },
    "et": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators":[300,600],
        "clf__max_depth":[None,12,18],
        "clf__min_samples_split":[2,4], 
        "clf__max_features":["sqrt","log2"]
    },
    "logreg": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__C":[0.25,0.5,1,2]
    },
    "gb": {
        "scaler":[StandardScaler(), RobustScaler(), "passthrough"],
        "clf__n_estimators":[200,400], 
        "clf__learning_rate":[0.03,0.05,0.1],
        "clf__max_depth":[2,3], 
        "clf__subsample":[0.8,1.0]
    },
}

scores = {}
best = {"name": None, "bal_acc": -np.inf, "est": None, "grid": None}

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
        return_train_score=True,
    )
    grid.fit(X_train_df, y_train, groups=g_train)

    cv_mean = grid.cv_results_["mean_test_score"][grid.best_index_]
    cv_std  = grid.cv_results_["std_test_score"][grid.best_index_]
    print(f"{name} — CV balanced acc: {cv_mean:.4f} ± {cv_std:.4f}")
    print("Best params:", grid.best_params_)

    y_pred = grid.predict(X_test_df)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    scores[name] = bal_acc
    print(f"{name} — TEST balanced acc: {bal_acc:.4f}")

    if bal_acc > best["bal_acc"]:
        best.update({"name": name, "bal_acc": bal_acc, "est": grid.best_estimator_, "grid": grid})

print("\nPer-model TEST balanced accuracy:", {k: f"{v:.4f}" for k, v in scores.items()})
print(f"Best model: {best['name']} | TEST balanced acc: {best['bal_acc']:.4f}")

best_est = best["est"]
y_hat = best_est.predict(X_test_df)
print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_hat))