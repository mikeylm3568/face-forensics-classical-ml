import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def get_classifiers():
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
            )),
        ]),
    }


def train_and_evaluate(X_base, X_enh, y, test_size=0.30, seed=42):
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=test_size, stratify=y, random_state=seed
    )

    results = []
    trained = {}

    for name, model in get_classifiers().items():
        model.fit(X_base[idx_train], y_train)
        p_base = model.predict(X_base[idx_test])
        results.append({
            "Feature Set": "Baseline (Color Hist, 96-D)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, p_base),
            "F1-Score": f1_score(y_test, p_base),
        })

        model.fit(X_enh[idx_train], y_train)
        p_enh = model.predict(X_enh[idx_test])
        trained[name] = (model, p_enh)
        results.append({
            "Feature Set": "Enhanced (FFT + Residual, 7-D)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, p_enh),
            "F1-Score": f1_score(y_test, p_enh),
        })

    return results, trained, (idx_train, idx_test, y_train, y_test)


def plot_confusion_matrices(trained, y_test, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, (_model, preds)) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    ax=ax)
        ax.set_title(f"{name} - Enhanced Features")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_misclassified(raw_images, idx_test, y_test, trained,
                       model_name="SVM (RBF)", max_show=10, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    model, preds = trained[model_name]
    wrong = np.where(preds != y_test)[0][:max_show]

    if len(wrong) == 0:
        print(f"[{model_name}] No misclassifications - nothing to plot.")
        return None

    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).ravel()

    for ax, i in zip(axes, wrong):
        img = raw_images[idx_test[i]]
        true_lbl = "Real" if y_test[i] == 0 else "Fake"
        pred_lbl = "Real" if preds[i] == 0 else "Fake"
        ax.imshow(img)
        ax.set_title(f"T:{true_lbl} / P:{pred_lbl}", fontsize=9, color="red")
        ax.axis("off")

    for ax in axes[len(wrong):]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "misclassified_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
