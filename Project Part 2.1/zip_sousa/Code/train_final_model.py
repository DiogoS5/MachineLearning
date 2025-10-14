import numpy as np
import pandas as pd
import pickle
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

def main():
    # Load data
    X_df = pd.read_pickle("Xtrain1.pkl")
    y = np.load("Ytrain1.npy")
    X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)
    assert X.shape[1] == 132, f"Expected 132 features, got {X.shape[1]}"

    # Build the final training pipeline using YOUR classes
    pipe = Pipeline([
        ("sanitize", Sanitize()),
        ("posenorm_tv", PoseNorm1(n_keypoints=NK)),                     # your torso-based normalizer
        ("mirror",     MirrorLeftToRightBody(n_keypoints=NK)),          # your mirroring step
        ("symmetry",   SymmetryFeatures(SYMM_PAIRS, n_keypoints=NK)),   # your symmetry extras
        ("scaler",     StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2, gamma="scale",
                    class_weight="balanced", probability=True, random_state=42)),
    ])

    print("Fitting SVC(RBF) on ALL data with current pipeline...")
    pipe.fit(X, y)
    print("Done.")

    # Optional quick sanity check on TRAIN set
    y_hat = pipe.predict(X)
    f1_tr = f1_score(y, y_hat, average="macro")
    print(f"Training F1_macro (on all data): {f1_tr:.4f}")

    # Save the trained pipeline
    out_path = "svc_rbf_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved final SVC(RBF) pipeline to: {out_path}")

if __name__ == "__main__":
    main()