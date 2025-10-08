# model_choosing.py — full script (plots only, saves nothing)
# - Uses ONLY Xtrain1.pkl and Ytrain1.npy
# - Patient-level outer split (4 patients for test), inner GroupKFold CV on train
# - Models: RF, ExtraTrees, GradientBoosting, SVM(RBF), Logistic, MLP (early stopping), kNN
# - GridSearchCV scoring='f1_macro'
# - Plots: per-model F1 bar, confusion matrix for best, MLP loss+val,
#          learning curve (grouped), permutation importance, PCA 2D scatter,
#          ROC & PR (if supported)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, GroupKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


X_df = pd.read_pickle("Xtrain1.pkl")
y = np.load("Ytrain1.npy")

X = np.stack(X_df["Skeleton_Features"].to_numpy()).astype(float)

patients = X_df["Patient_Id"].to_numpy()
unique_patients = np.unique(patients)

train_patients, test_patients = train_test_split(unique_patients, test_size=4, random_state=42)
print("Train patients:", np.sort(train_patients))
print("Test patients :", np.sort(test_patients))

train_mask = np.isin(patients, train_patients)
test_mask  = np.isin(patients, test_patients)

X_train, y_train, g_train = X[train_mask], y[train_mask], patients[train_mask]
X_test,  y_test,  g_test  = X[test_mask],  y[test_mask],  patients[test_mask]

pipelines = {
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "extratrees": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(random_state=42, n_jobs=-1))
    ]),
    "gb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    "svc_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    ]),
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42))
    ]),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            alpha=1e-4,
            early_stopping=True,
            n_iter_no_change=20,
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
    "gb": {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [1.0, 0.8],
    },
    "svc_rbf": {
        "clf__C": [0.5, 1, 2, 4],
        "clf__gamma": ["scale", 0.05, 0.02, 0.01],
    },
    "logreg": {
        "clf__C": [0.5, 1, 2, 4]
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3],
    },
    "knn": {
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
    grid.fit(X_train, y_train, **({"groups": g_train}))

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

# ------------------------------
# Learning curve for the best model (GroupKFold on TRAIN)
# ------------------------------
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

# ------------------------------
# Permutation importance (on TEST)
# ------------------------------
try:
    perm = permutation_importance(
        best_model, X_test, y_test, scoring="f1_macro",
        n_repeats=10, random_state=42, n_jobs=-1
    )
    imp_idx = np.argsort(perm.importances_mean)[::-1][:15]
    plt.figure(figsize=(7,5))
    plt.barh(range(len(imp_idx)), perm.importances_mean[imp_idx][::-1], xerr=perm.importances_std[imp_idx][::-1])
    plt.yticks(range(len(imp_idx)), [f"f{j}" for j in imp_idx[::-1]])
    plt.xlabel("Mean ΔF1 (permutation)")
    plt.title("Top-15 Permutation Importances (TEST)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Permutation importance failed:", e)

# ------------------------------
# PCA (fit on TRAIN), scatter whole dataset for visualization
# ------------------------------
scaler_for_pca = StandardScaler().fit(X_train)
X_train_std = scaler_for_pca.transform(X_train)
X_test_std  = scaler_for_pca.transform(X_test)

pca = PCA(n_components=2, random_state=42).fit(X_train_std)
X_all_2d = pca.transform(scaler_for_pca.transform(X))

plt.figure(figsize=(6.2,5.2))
classes = np.sort(np.unique(y))
for c in classes:
    plt.scatter(X_all_2d[y == c, 0], X_all_2d[y == c, 1], alpha=0.6, label=f"Class {c}", s=18)
plt.legend(title="Class")
plt.title("PCA (2D) of Skeleton Features (PCA fit on TRAIN)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# ------------------------------
# ROC & PR curves (One-vs-Rest) — only if model supports probabilities/scores
# ------------------------------
def get_scores_for_ovr(model, X_data):
    # Prefer decision_function (SVM margins), else predict_proba
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_data)
        # shape (n_samples, n_classes) expected; if 1D, convert
        if s.ndim == 1:
            s = np.column_stack([1 - s, s])
        return s
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)
    return None

scores_mat = get_scores_for_ovr(best_model, X_test)
if scores_mat is not None:
    classes = np.sort(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=classes)

    # ROC
    plt.figure(figsize=(6.2,5.2))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores_mat[:, i])
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curves — {best_name} (TEST)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PR
    plt.figure(figsize=(6.2,5.2))
    ap_scores = []
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], scores_mat[:, i])
        ap = average_precision_score(y_test_bin[:, i], scores_mat[:, i])
        ap_scores.append(ap)
        plt.plot(rec, prec, label=f"Class {c} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves — {best_name} (TEST)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"ROC/PR skipped for {best_name}: model has neither decision_function nor predict_proba.")
