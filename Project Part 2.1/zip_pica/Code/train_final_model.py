import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
SYMM_PAIRS = [(L_SH, R_SH), (L_EL, R_EL), (L_WR, R_WR)]
NK = 33  # number of keypoints

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

class PoseNorm(BaseEstimator, TransformerMixin):
    def __init__(self, n_keypoints=NK, eps=1e-9):
        self.n_keypoints = n_keypoints
        self.eps = eps
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        N = X.shape[0]
        nk = self.n_keypoints
        means = X[:, :2*nk].reshape(N, nk, 2)
        stds  = X[:, 2*nk:].reshape(N, nk, 2)

        centroid = means.mean(axis=1, keepdims=True)
        m = means - centroid

        size = np.sqrt((m**2).sum(axis=(1,2)) / nk).reshape(N, 1, 1)
        size = np.maximum(size, self.eps)
        m /= size
        stds /= size

        m_rot = np.empty_like(m)
        s_rot = np.empty_like(stds)
        for i in range(N):
            C = (m[i].T @ m[i]) / nk
            _, V = np.linalg.eigh(C)
            R = V[:, [1, 0]]
            if np.linalg.det(R) < 0:
                R[:, 1] *= -1
            m_rot[i] = m[i] @ R
            s_rot[i] = stds[i] @ R

        return np.concatenate([m_rot.reshape(N, -1), s_rot.reshape(N, -1)], axis=1)

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

def main():
    X_df = pd.read_pickle("Xtrain1.pkl")
    y = np.load("Ytrain1.npy")

    X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)
    assert X.shape[1] == 132, f"Expected 132 features, got {X.shape[1]}"

    pipe = Pipeline([
        ("sanitize", Sanitize()),
        ("posenorm", PoseNorm(n_keypoints=NK)),
        ("symmetry", SymmetryFeatures(SYMM_PAIRS, n_keypoints=NK)),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2, gamma="scale",
                    class_weight="balanced", probability=True, random_state=42)),
    ])

    print("Fitting svc_rbf on ALL data with best params...")
    pipe.fit(X, y)
    print("Done.")

    # Optional: quick sanity check
    y_hat = pipe.predict(X)
    f1_tr = f1_score(y, y_hat, average="macro")
    print(f"Training F1_macro (on all data): {f1_tr:.4f}")

    # --- Save model ---
    out_path = "svc_rbf_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved final SVC(RBF) pipeline to: {out_path}")

if __name__ == "__main__":
    main()
