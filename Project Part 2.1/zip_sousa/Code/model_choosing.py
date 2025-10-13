import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, TransformerMixin

L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
SYMM_PAIRS = [(L_SH, R_SH), (L_EL, R_EL), (L_WR, R_WR)]
NK = 33

class Sanitize(BaseEstimator, TransformerMixin):
    def __init__(self, q_low=0.001, q_high=0.999):
        self.q_low = q_low
        self.q_high = q_high
        
    def fit(self, X, y=None):
        Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.lo_ = np.quantile(Xf, self.q_low, axis=0)
        self.hi_ = np.quantile(Xf, self.q_high, axis=0)
        return self
    
    def transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, self.lo_, self.hi_)


class PoseNorm1(BaseEstimator, TransformerMixin):
    def __init__(self, n_keypoints=NK, eps=1e-9):
        self.n_keypoints = n_keypoints
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples  = X.shape[0]
        n_keypoints = self.n_keypoints

        # Split into mean and std coordinates
        mean_coords = X[:, :2 * n_keypoints].reshape(n_samples, n_keypoints, 2)
        std_coords  = X[:, 2 * n_keypoints:].reshape(n_samples, n_keypoints, 2)

        # Indices for torso/head distances
        shoulders = [11, 12]
        hips = [23, 24]
        head_idx = 0
        torso_indices = shoulders + hips  # [11,12,23,24]

        torso_centroid = mean_coords[:, torso_indices].mean(axis=1)  # (n_samples, 2)

        shoulder_distance = np.linalg.norm(
            mean_coords[:, shoulders[0]] - mean_coords[:, shoulders[1]], axis=1
        )  # (n_samples,)

        # Head-to-mid-hip distance (vertical scale)
        mid_hip = mean_coords[:, hips].mean(axis=1)  # (n_samples, 2)
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

        # Flatten back to (n_samples, 132)
        return np.concatenate(
            [mean_normalized.reshape(n_samples, -1),
             std_normalized.reshape(n_samples, -1)],
            axis=1
        )

class SymmetryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, pairs, n_keypoints=33):
        self.pairs = pairs
        self.n_keypoints = n_keypoints
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        N = X.shape[0]
        nk = self.n_keypoints
        means = X[:, :2*nk].reshape(N, nk, 2)
        extras = []
        
        for (L, R) in self.pairs:
            sym_mag = np.linalg.norm(means[:, L, :] - means[:, R, :], axis=1, keepdims=True)
            extras.append(sym_mag)
            
        if extras:
            return np.concatenate([X, np.hstack(extras)], axis=1)
        return X

# For side detection and mirroring
left_arm  = [13, 15, 17, 19, 21]
right_arm = [14, 16, 18, 20, 22]

# Left/right index swaps (include head/face)
LEFT_POINTS  = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RIGHT_POINTS = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

def detect_mov_side(feats_132):
    """Heuristic: compare summed stds of left vs right arm keypoints."""
    stds = feats_132[66:].reshape(33, 2)
    left_std_sum  = np.sum(stds[left_arm])
    right_std_sum = np.sum(stds[right_arm])
    return 'left' if left_std_sum >= right_std_sum else 'right'

class MirrorLeftToRightBody(BaseEstimator, TransformerMixin):
    """Mirror samples detected as LEFT to RIGHT (reflect x about torso-centroid; swap L/R indices)."""
    def __init__(self, n_keypoints=NK, enable=True):
        self.n_keypoints = n_keypoints
        self.enable = enable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.enable:
            return X
        N  = X.shape[0]
        nk = self.n_keypoints
        X_out = np.empty_like(X)
        for i in range(N):
            feats = X[i].copy()
            means = feats[:2*nk].reshape(nk, 2)
            stds  = feats[2*nk:].reshape(nk, 2)

            side = detect_mov_side(feats)
            if side == "left":
                torso_indices = [11, 12, 23, 24]
                centroid = means[torso_indices].mean(axis=0)

                means_m = means.copy()
                stds_m  = stds.copy()

                means_m[:, 0] = 2 * centroid[0] - means[:, 0]

                for l, r in zip(LEFT_POINTS, RIGHT_POINTS):
                    means_m[l], means_m[r] = means_m[r].copy(), means_m[l].copy()
                    stds_m[l],  stds_m[r]  = stds_m[r].copy(),  stds_m[l].copy()

                means, stds = means_m, stds_m

            X_out[i] = np.concatenate([means.flatten(), stds.flatten()])
        return X_out

X_df = pd.read_pickle("Xtrain1.pkl")
y = np.load("Ytrain1.npy")

X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)
assert X.shape[1] == 132, f"Expected 132 features, got {X.shape[1]}"

patients = X_df["Patient_Id"].to_numpy()
unique_patients = np.unique(patients)

train_patients, test_patients = train_test_split(unique_patients, test_size=4, random_state=42)
print("Train patients:", np.sort(train_patients))
print("Test patients :", np.sort(test_patients))

train_mask = np.isin(patients, train_patients)
test_mask  = np.isin(patients, test_patients)

X_train, y_train, g_train = X[train_mask], y[train_mask], patients[train_mask]
X_test,  y_test,  g_test  = X[test_mask],  y[test_mask],  patients[test_mask]

pre_base = [
    ("sanitize", Sanitize()),
    ("posenorm_tv", PoseNorm1(n_keypoints=NK)),
    ("mirror", MirrorLeftToRightBody(n_keypoints=NK)),
    ("symmetry",  SymmetryFeatures(SYMM_PAIRS, n_keypoints=NK)),
]

pipelines = {
    "rf": Pipeline(pre_base + [
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "svc_rbf": Pipeline(pre_base + [
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    ]),
    "mlp": Pipeline(pre_base + [
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            alpha=1e-4,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42))
    ]),
    "knn": Pipeline(pre_base + [
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
}

param_grids = {
    "rf": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
    },
    "svc_rbf": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__C": [0.5, 1, 2, 4],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", 0.05, 0.02, 0.01],
    },
    "mlp": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3],
    },
    "knn": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
}

inner_cv = GroupKFold(n_splits=5)

scores = {}
best_model = None
best_name = None
best_test_f1 = -np.inf
mlp_loss_curve = None
mlp_val_scores = None

for name, pipe in pipelines.items():
    print(f"\nTraining {name} with GridSearchCV (GroupKFold, scoring='f1_macro')...")
    
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids.get(name, {}),
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_train, y_train, groups=g_train)

    y_pred_test = grid.predict(X_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")
    scores[name] = f1

    print(f"{name} — Test F1_macro: {f1:.4f}")
    print("Best params:", grid.best_params_)

    if name == "mlp":
        mlp_est = grid.best_estimator_.named_steps["clf"]
        if hasattr(mlp_est, "loss_curve_"):
            mlp_loss_curve = mlp_est.loss_curve_
        if hasattr(mlp_est, "validation_scores_"):
            mlp_val_scores = mlp_est.validation_scores_

    if f1 > best_test_f1:
        best_test_f1 = f1
        best_model = grid.best_estimator_
        best_name = name
        best_grid = grid

print(f"\nBest model: {best_name} | Test F1_macro: {best_test_f1:.4f}\n")


plt.figure(figsize=(7,4))
names, vals = zip(*sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
plt.bar(names, vals)
plt.ylabel("Test F1 (macro)")
plt.title("Per-model F1 on held-out patients")
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

y_pred_best = best_model.predict(X_test)
labels_sorted = np.sort(np.unique(y))
cm = confusion_matrix(y_test, y_pred_best, labels=labels_sorted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
fig, ax = plt.subplots(figsize=(5.5,5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — {best_name} (TEST)")
plt.tight_layout()
plt.show()

if (mlp_loss_curve is not None) and (mlp_val_scores is not None):
    n = min(len(mlp_loss_curve), len(mlp_val_scores))
    epochs = np.arange(1, n+1)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, mlp_loss_curve[:n], label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, mlp_val_scores[:n], linestyle="--", label="Validation score")
    ax2.set_ylabel("Validation score")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="best")
    plt.title("MLP: Training Loss & Validation Score vs Epoch")
    plt.tight_layout()
    plt.show()
elif mlp_loss_curve is not None:
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(mlp_loss_curve)+1), mlp_loss_curve)
    plt.xlabel("Epoch"); plt.ylabel("Training loss")
    plt.title("MLP Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    groups=g_train,
    cv=inner_cv,
    scoring="f1_macro",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    shuffle=True,
    random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(7,4))
plt.plot(train_sizes, train_mean, marker="o", label="Train (cv mean)")
plt.plot(train_sizes, val_mean, marker="s", label="Validation (cv mean)")
plt.xlabel("Training samples")
plt.ylabel("F1 (macro)")
plt.title(f"Learning Curve — {best_name}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

preprocess_for_pca = Pipeline(pre_base + [
    ("scaler_before_pca", StandardScaler())
])

# Fit preprocessing on TRAIN only, then transform ALL for plotting
preprocess_for_pca.fit(X_train, y_train)
X_all_std   = preprocess_for_pca.transform(X)
X_train_std = X_all_std[train_mask]
X_test_std  = X_all_std[test_mask]

pca = PCA(n_components=2, random_state=42).fit(X_train_std)
X_all_2d = pca.transform(X_all_std)

# ---- Plot ----
plt.figure(figsize=(6.2, 5.2))
classes = np.sort(np.unique(y))
for c in classes:
    pts = X_all_2d[y == c]
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f"Class {c}", s=18)

evr = pca.explained_variance_ratio_ * 100
plt.legend(title="Class")
plt.title("PCA (2D) of Pipeline-Transformed Features (PoseNorm → Scale; PCA fit on TRAIN)")
plt.xlabel(f"PC1 ({evr[0]:.1f}%)")
plt.ylabel(f"PC2 ({evr[1]:.1f}%)")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
