# model_choosing.py — full script (plots only, saves nothing)
# - Uses ONLY Xtrain1.pkl and Ytrain1.npy
# - Models: RF, ExtraTrees, GradientBoosting, SVM(RBF), Logistic, MLP (early stopping), kNN
# - GridSearchCV scoring='f1_macro'
# - Plots: per-model F1 bar, confusion matrix for best, MLP loss+val on same graph,
#          learning curve, feature/permutation importance, PCA 2D scatter, ROC & PR (if supported)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


X_df = pd.read_pickle("Xtrain1.pkl")
y = np.load("Ytrain1.npy")

X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)

assert X.ndim == 2 and X.shape[1] == 132, f"Expected (n,132), got {X.shape}"
assert X.shape[0] == y.shape[0], "X and y sample counts differ"

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipelines = {
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "extratrees": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(random_state=42, n_jobs=-1))
    ]),
    "gboost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    "svc_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ]),
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42))
    ]),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,          # more iterations to see the curve
            alpha=1e-4,
            early_stopping=True,    # enables validation_scores_
            n_iter_no_change=20,    # patience for early stopping
            random_state=42))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
}

param_grids = {
    "rf": {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
    },
    "extratrees": {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 12, 18],
        "clf__min_samples_split": [2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "svc_rbf": {
        "clf__C": [0.5, 1, 2, 4],
        "clf__gamma": ["scale", 0.05, 0.02, 0.01],
    },
    "logreg": {
        "clf__C": [0.5, 1, 2],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],  # 1=Manhattan, 2=Euclidean
    },
}


scores = {}
best_model = None
best_name = None
best_val_f1 = -np.inf

mlp_loss_curve = None
mlp_val_scores = None

for name, pipe in pipelines.items():
    print(f"\n🔹 Training {name} with GridSearchCV (scoring='f1_macro')...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids.get(name, {}),
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
        refit=True
    )
    grid.fit(X_train, y_train)

    y_pred_val = grid.predict(X_val)
    f1 = f1_score(y_val, y_pred_val, average="macro")
    scores[name] = f1

    print(f"{name} — Validation F1_macro: {f1:.4f}")
    print("Best params:", grid.best_params_)

    if name == "mlp":
        mlp_est = grid.best_estimator_.named_steps["clf"]
        if hasattr(mlp_est, "loss_curve_"):
            mlp_loss_curve = mlp_est.loss_curve_
        if hasattr(mlp_est, "validation_scores_"):
            mlp_val_scores = mlp_est.validation_scores_

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_model = grid.best_estimator_
        best_name = name

print(f"\nBest model: {best_name} | F1_macro (val): {best_val_f1:.4f}")

if (mlp_loss_curve is not None) and (mlp_val_scores is not None):
    n = min(len(mlp_loss_curve), len(mlp_val_scores))
    epochs = np.arange(1, n + 1)

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
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title("MLP: Training Loss & Validation Score vs Epoch")
    plt.tight_layout()
    plt.show()
elif mlp_loss_curve is not None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(mlp_loss_curve)+1), mlp_loss_curve)
    plt.xlabel("Epoch"); plt.ylabel("Training loss")
    plt.title("MLP Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


feat_names = feature_names()
clf = best_model.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feat_names)[idx], importances[idx])
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.tight_layout()
    plt.show()
else:
    try:
        r = permutation_importance(best_model, X_val, y_val, scoring="f1_macro", n_repeats=10, n_jobs=-1)
        idx = np.argsort(r.importances_mean)[-15:]
        plt.figure(figsize=(8, 5))
        plt.barh(np.array(feat_names)[idx], r.importances_mean[idx])
        plt.title(f"Top 15 Permutation Importances — {best_name}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Permutation importance failed:", e)


X2 = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(6, 5))
for lab, name in zip([0, 1, 2], ["E1", "E2", "E5"]):
    plt.scatter(X2[y == lab, 0], X2[y == lab, 1], alpha=0.6, label=name)
plt.legend()
plt.title("PCA (2D) of Skeleton Features")
plt.tight_layout()
plt.show()