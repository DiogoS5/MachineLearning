# eda_clusters_and_movement_readable.py
# - PCA & t-SNE with readable class names
# - Movement (std magnitude) for hips/shoulders/hands with readable labels
# - No files saved

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# 1) Load data
# -----------------------------
X_df = pd.read_pickle("Xtrain1.pkl")
y = np.load("Ytrain1.npy")

if "Skeleton_Features" in X_df.columns:
    X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)
elif all(np.issubdtype(dt, np.number) for dt in X_df.dtypes) and X_df.shape[1] == 132:
    X = X_df.to_numpy(dtype=float)
else:
    raise ValueError("Expected 'Skeleton_Features' column (len 132 arrays) or 132 numeric columns.")

assert X.shape[0] == y.shape[0] and X.shape[1] == 132

# Friendly class names
CLASS_NAME = {
    0: "Brushing Hair (E1)",
    1: "Brushing Teeth (E2)",
    2: "Hip Flexion (E5)",
}
y_txt = np.vectorize(CLASS_NAME.get)(y.astype(int))

# -----------------------------
# 2) Helpers
# -----------------------------
def std_mag_for_kp(Xmat, kp):
    """sqrt(std_x^2 + std_y^2) for keypoint kp from feature matrix X (n,132)."""
    sx = 66 + 2*kp
    sy = 66 + 2*kp + 1
    return np.sqrt(Xmat[:, sx]**2 + Xmat[:, sy]**2)

def std_mag_for_group(Xmat, kps):
    mags = np.column_stack([std_mag_for_kp(Xmat, k) for k in kps])
    return mags.mean(axis=1)

# MediaPipe keypoints of interest
KP = {
    "L_shoulder": 11, "R_shoulder": 12,
    "L_hip": 23, "R_hip": 24,
    "L_wrist": 15, "R_wrist": 16,
    "L_hand_set": [15, 17, 19, 21],  # wrist, pinky, index, thumb (left)
    "R_hand_set": [16, 18, 20, 22],  # wrist, pinky, index, thumb (right)
}

# -----------------------------
# 3) Standardize for embeddings
# -----------------------------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# -----------------------------
# 4) PCA (2D) scatter with explained variance (readable names)
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)
evr = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(6.8, 5.6))
for cls in [0, 1, 2]:
    pts = X_pca[y == cls]
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.75, label=CLASS_NAME[cls])
plt.xlabel(f"PC1 ({evr[0]:.1f}%)")
plt.ylabel(f"PC2 ({evr[1]:.1f}%)")
plt.title("PCA (2D): Class Clusters")
plt.legend(title="Exercise")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) t-SNE (2D) scatter (readable names)
# -----------------------------
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_std)

plt.figure(figsize=(6.8, 5.6))
for cls in [0, 1, 2]:
    pts = X_tsne[y == cls]
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.75, label=CLASS_NAME[cls])
plt.title("t-SNE (2D): Class Clusters")
plt.legend(title="Exercise")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Movement metrics (std magnitudes) with readable region names
# -----------------------------
df_move = pd.DataFrame({
    "Exercise": y_txt,
    "Left Shoulder (std)":  std_mag_for_kp(X, KP["L_shoulder"]),
    "Right Shoulder (std)": std_mag_for_kp(X, KP["R_shoulder"]),
    "Left Hip (std)":       std_mag_for_kp(X, KP["L_hip"]),
    "Right Hip (std)":      std_mag_for_kp(X, KP["R_hip"]),
    "Left Wrist (std)":     std_mag_for_kp(X, KP["L_wrist"]),
    "Right Wrist (std)":    std_mag_for_kp(X, KP["R_wrist"]),
    "Left Hand (avg std)":  std_mag_for_group(X, KP["L_hand_set"]),
    "Right Hand (avg std)": std_mag_for_group(X, KP["R_hand_set"]),
})

# A) Class-wise boxplots for key regions (matplotlib-only)
melt = df_move.melt(id_vars="Exercise", var_name="Region", value_name="Std Magnitude")

regions = melt["Region"].unique().tolist()
xpos = np.arange(len(regions))
width = 0.25

plt.figure(figsize=(12, 6))
for i, ex in enumerate([CLASS_NAME[0], CLASS_NAME[1], CLASS_NAME[2]]):
    meds = [np.median(melt[(melt["Region"] == r) & (melt["Exercise"] == ex)]["Std Magnitude"])
            for r in regions]
    plt.bar(xpos + (i - 1)*width, meds, width=width, alpha=0.8, label=ex)
plt.xticks(xpos, regions, rotation=45, ha="right")
plt.ylabel("Std Magnitude")
plt.title("Movement by Region (std magnitude) — Class-wise Medians")
plt.legend(title="Exercise")
plt.tight_layout()
plt.show()

# B) Left vs Right scatter for shoulders / hips / hands (asymmetry view, readable axes)
pairs = [
    ("Left Shoulder (std)",  "Right Shoulder (std)",  "Shoulders: Left vs Right"),
    ("Left Hip (std)",       "Right Hip (std)",       "Hips: Left vs Right"),
    ("Left Hand (avg std)",  "Right Hand (avg std)",  "Hands: Left vs Right (avg std)"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (lx, rx, title) in zip(axes, pairs):
    for ex in [CLASS_NAME[0], CLASS_NAME[1], CLASS_NAME[2]]:
        m = df_move["Exercise"] == ex
        ax.scatter(df_move.loc[m, lx], df_move.loc[m, rx], alpha=0.75, label=ex)
    mn = float(min(df_move[lx].min(), df_move[rx].min()))
    mx = float(max(df_move[lx].max(), df_move[rx].max()))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4)
    ax.set_xlabel(lx); ax.set_ylabel(rx)
    ax.set_title(title); ax.grid(True, alpha=0.3)

# single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", title="Exercise")
plt.tight_layout()
plt.show()

print("EDA complete: PCA/t-SNE clustering + movement (hips/shoulders/hands) with readable names.")
