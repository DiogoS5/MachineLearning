# eda_clusters_and_movement_readable.py
# 2D linear (PCA) & optional non-linear (t-SNE) projections + movement summaries

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_PKL = "Xtrain1.pkl"
LABEL_NPY = "Ytrain1.npy"
N_FEATURES = 132

CLASS_NAME = {
    0: "Brushing Hair (E1)",
    1: "Brushing Teeth (E2)",
    2: "Hip Flexion (E5)",
}

KP = {
    "L_shoulder": 11, "R_shoulder": 12,
    "L_hip": 23, "R_hip": 24,
    "L_wrist": 15, "R_wrist": 16,
    "L_hand_set": [15, 17, 19, 21],
    "R_hand_set": [16, 18, 20, 22],
}

X_df = pd.read_pickle(DATA_PKL)
y = np.load(LABEL_NPY)
X = np.vstack(X_df["Skeleton_Features"].values).astype(float)

if X.shape[0] != len(y) or X.shape[1] != N_FEATURES:
    raise ValueError(f"Bad shapes: X={X.shape}, y={len(y)} (expected features={N_FEATURES})")

y_txt = np.vectorize(CLASS_NAME.get)(y.astype(int))


def kp_cols(kp_idx: int):
    """Return column indices (x,y) for a keypoint kp_idx under your 132-feature layout."""
    x_col = 66 + 2 * kp_idx
    y_col = x_col + 1
    return x_col, y_col

def mag_for_kp(Xmat: np.ndarray, kp_idx: int) -> np.ndarray:
    """Magnitude from the two columns for a keypoint."""
    cx, cy = kp_cols(kp_idx)
    return np.sqrt(Xmat[:, cx]**2 + Xmat[:, cy]**2)

def mag_for_group(Xmat: np.ndarray, kps: list[int]) -> np.ndarray:
    """Average magnitude over a list of keypoints."""
    return np.column_stack([mag_for_kp(Xmat, k) for k in kps]).mean(axis=1)

def scatter_by_class(Z2: np.ndarray, y: np.ndarray, title: str, xlabel=None, ylabel=None):
    plt.figure(figsize=(6.8, 5.6))
    for cls in sorted(np.unique(y)):
        pts = Z2[y == cls]
        plt.scatter(pts[:, 0], pts[:, 1], alpha=0.75, label=CLASS_NAME[cls])
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Exercise")
    plt.tight_layout()
    plt.show()


X_std = StandardScaler().fit_transform(X)

linear_proj = PCA(n_components=2, random_state=42)
X_lin = linear_proj.fit_transform(X_std)
expl = linear_proj.explained_variance_ratio_ * 100

scatter_by_class(
    X_lin, y,
    title="2D Linear Projection: Class Clusters",
    xlabel=f"Component 1 ({expl[0]:.1f}%)",
    ylabel=f"Component 2 ({expl[1]:.1f}%)",
)


df_move = pd.DataFrame({
    "Exercise": y_txt,
    "Left Shoulder":  mag_for_kp(X, KP["L_shoulder"]),
    "Right Shoulder": mag_for_kp(X, KP["R_shoulder"]),
    "Left Hip":       mag_for_kp(X, KP["L_hip"]),
    "Right Hip":      mag_for_kp(X, KP["R_hip"]),
    "Left Wrist":     mag_for_kp(X, KP["L_wrist"]),
    "Right Wrist":    mag_for_kp(X, KP["R_wrist"]),
    "Left Hand":      mag_for_group(X, KP["L_hand_set"]),
    "Right Hand":     mag_for_group(X, KP["R_hand_set"]),
})

melt = df_move.melt(id_vars="Exercise", var_name="Region", value_name="Magnitude")
regions = melt["Region"].unique().tolist()
xpos = np.arange(len(regions))
width = 0.25

plt.figure(figsize=(12, 6))
for i, cls_name in enumerate([CLASS_NAME[k] for k in sorted(CLASS_NAME)]):
    meds = [np.median(melt[(melt["Region"] == r) & (melt["Exercise"] == cls_name)]["Magnitude"])
            for r in regions]
    plt.bar(xpos + (i - 1)*width, meds, width=width, alpha=0.8, label=cls_name)
plt.xticks(xpos, regions, rotation=45, ha="right")
plt.ylabel("Magnitude")
plt.title("Movement by Region — Class-wise Medians")
plt.legend(title="Exercise")
plt.tight_layout()
plt.show()

pairs = [
    ("Left Shoulder",  "Right Shoulder",  "Shoulders: Left vs Right"),
    ("Left Hip",       "Right Hip",       "Hips: Left vs Right"),
    ("Left Hand",      "Right Hand",      "Hands: Left vs Right"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (lx, rx, title) in zip(axes, pairs):
    for cls_name in [CLASS_NAME[k] for k in sorted(CLASS_NAME)]:
        m = df_move["Exercise"] == cls_name
        ax.scatter(df_move.loc[m, lx], df_move.loc[m, rx], alpha=0.75, label=cls_name)
    mn = float(min(df_move[lx].min(), df_move[rx].min()))
    mx = float(max(df_move[lx].max(), df_move[rx].max()))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4)
    ax.set_xlabel(lx); ax.set_ylabel(rx)
    ax.set_title(title); ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", title="Exercise")
plt.tight_layout()
plt.show()

print("EDA complete.")