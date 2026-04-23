import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SVM_FEATURES = [
    "daily_usage_hours",
    "late_night_hours",
    "comparison_content_pct",
    "fomo_score",
    "sessions_per_day",
    "sleep_quality_score",
    "self_esteem_score",
    "anxiety_score",
    "life_satisfaction",
    "loneliness_score",
    "notifications_per_day",
    "avg_session_duration_min",
    "platforms_used",
    "posts_per_week",
    "stories_per_week",
    "likes_received_weekly",
    "comments_received_weekly",
    "engagement_ratio",
]


def get_svm_features(df: pd.DataFrame):
    features = [feature for feature in SVM_FEATURES if feature in df.columns]
    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    return X, features


def prepare_svm_target(df: pd.DataFrame):
    return (df["depression_score"] > 14).astype(int)


def prepare_svm_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X, feature_names = get_svm_features(df)
    y = prepare_svm_target(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_train_pca2 = pca_2d.fit_transform(X_train_scaled)
    X_test_pca2 = pca_2d.transform(X_test_scaled)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "pca_2d": pca_2d,
        "X_train_pca2": X_train_pca2,
        "X_test_pca2": X_test_pca2,
    }


def _build_svc(kernel: str, C: float):
    params = {
        "kernel": kernel,
        "C": C,
        "decision_function_shape": "ovr"
    }
    if kernel == "poly":
        params.update({"degree": 3, "coef0": 1})
    return SVC(**params)


def run_svm_experiments(prep: dict, kernels=None, costs=None):
    if kernels is None:
        kernels = ("linear", "poly", "rbf")
    if costs is None:
        costs = (0.001, 0.01, 0.1, 1, 10, 100)

    results = {}
    labels = [0, 1]

    for kernel in kernels:
        trials = []
        for cost in costs:
            model = _build_svc(kernel, cost)
            model.fit(prep["X_train_scaled"], prep["y_train"])
            preds = model.predict(prep["X_test_scaled"])
            trial = {
                "kernel": kernel,
                "C": float(cost),
                "model": model,
                "preds": preds,
                "accuracy": accuracy_score(prep["y_test"], preds),
                "confusion_matrix": confusion_matrix(prep["y_test"], preds, labels=labels),
                "labels": labels,
                "support_vectors": int(model.n_support_.sum()),
            }
            trials.append(trial)

        best_trial = sorted(trials, key=lambda item: (-item["accuracy"], item["C"]))[0]
        results[kernel] = {
            "trials": trials,
            "best": best_trial
        }

    return results


def svm_accuracy_table(results: dict):
    rows = []
    for kernel, details in results.items():
        for trial in details["trials"]:
            rows.append({
                "Kernel": kernel,
                "Cost (C)": trial["C"],
                "Accuracy": round(trial["accuracy"], 4),
                "Support Vectors": trial["support_vectors"],
            })
    return pd.DataFrame(rows).sort_values(["Kernel", "Cost (C)"]).reset_index(drop=True)


def best_svm_table(results: dict):
    rows = []
    for kernel, details in results.items():
        best = details["best"]
        rows.append({
            "Kernel": kernel,
            "Best Cost (C)": best["C"],
            "Accuracy": round(best["accuracy"], 4),
            "Support Vectors": best["support_vectors"],
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(cm, labels=("Lower Risk", "Higher Risk"), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_accuracy_by_cost(results: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {
        "linear": "#4C78A8",
        "poly": "#F58518",
        "rbf": "#54A24B",
    }
    for kernel, details in results.items():
        trials = sorted(details["trials"], key=lambda item: item["C"])
        costs = [trial["C"] for trial in trials]
        accuracies = [trial["accuracy"] for trial in trials]
        ax.plot(costs, accuracies, marker="o", linewidth=2, label=kernel.upper(), color=palette.get(kernel))

    ax.set_xscale("log")
    ax.set_xlabel("Cost (C)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("SVM Accuracy Across Cost Values", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


def plot_svm_margin_concept():
    x1 = np.array([1.0, 1.3, 1.8, 2.1, 2.6, 2.9])
    y1 = np.array([1.0, 1.6, 1.3, 2.1, 1.8, 2.6])
    x2 = np.array([4.2, 4.7, 5.1, 5.4, 5.9, 6.2])
    y2 = np.array([4.3, 5.0, 4.6, 5.5, 4.9, 5.8])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x1, y1, c="#4C78A8", s=90, label="Class A")
    ax.scatter(x2, y2, c="#E45756", s=90, label="Class B")

    line_x = np.linspace(0.5, 6.6, 200)
    center_y = line_x
    margin_up = line_x + 0.9
    margin_down = line_x - 0.9

    ax.plot(line_x, center_y, color="black", linewidth=2.5, label="Separating hyperplane")
    ax.plot(line_x, margin_up, color="black", linestyle="--", alpha=0.75)
    ax.plot(line_x, margin_down, color="black", linestyle="--", alpha=0.75)

    support_points = np.array([[2.1, 2.1], [4.2, 4.3]])
    ax.scatter(
        support_points[:, 0],
        support_points[:, 1],
        s=240,
        facecolors="none",
        edgecolors="black",
        linewidth=2,
        label="Support vectors"
    )

    ax.set_title("SVM Margin and Support Vectors", fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_kernel_trick_concept():
    angles = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    inner = np.column_stack((0.9 * np.cos(angles), 0.9 * np.sin(angles)))
    outer = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles)))

    fig = plt.figure(figsize=(14, 5.5))

    ax1 = fig.add_subplot(121)
    ax1.scatter(inner[:, 0], inner[:, 1], c="#4C78A8", s=55, label="Inner ring")
    ax1.scatter(outer[:, 0], outer[:, 1], c="#F58518", s=55, label="Outer ring")
    ax1.set_title("Original 2D Space", fontweight="bold")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(122, projection="3d")
    inner_z = np.sum(inner ** 2, axis=1)
    outer_z = np.sum(outer ** 2, axis=1)
    ax2.scatter(inner[:, 0], inner[:, 1], inner_z, c="#4C78A8", s=40)
    ax2.scatter(outer[:, 0], outer[:, 1], outer_z, c="#F58518", s=40)
    ax2.plot(
        [-2.5, 2.5, 2.5, -2.5, -2.5],
        [-2.5, -2.5, 2.5, 2.5, -2.5],
        [2.5, 2.5, 2.5, 2.5, 2.5],
        color="black",
        linewidth=2,
        alpha=0.8
    )
    ax2.set_title("Lifted Space: A Flat Split Becomes Possible", fontweight="bold")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("x1^2 + x2^2")

    plt.tight_layout()
    return fig


def polynomial_feature_cast_example(x1=2, x2=3):
    sqrt2 = np.sqrt(2)
    cast = pd.DataFrame(
        [
            {"Expanded feature": "x1^2", "Value": x1 ** 2},
            {"Expanded feature": "sqrt(2) * x1 * x2", "Value": round(sqrt2 * x1 * x2, 4)},
            {"Expanded feature": "x2^2", "Value": x2 ** 2},
            {"Expanded feature": "sqrt(2) * x1", "Value": round(sqrt2 * x1, 4)},
            {"Expanded feature": "sqrt(2) * x2", "Value": round(sqrt2 * x2, 4)},
            {"Expanded feature": "1", "Value": 1},
        ]
    )
    return cast


def plot_pca_decision_regions(prep: dict, kernel: str, C: float):
    model = _build_svc(kernel, C)
    model.fit(prep["X_train_pca2"], prep["y_train"])

    x_min, x_max = prep["X_train_pca2"][:, 0].min() - 1.0, prep["X_train_pca2"][:, 0].max() + 1.0
    y_min, y_max = prep["X_train_pca2"][:, 1].min() - 1.0, prep["X_train_pca2"][:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.contourf(xx, yy, zz, alpha=0.2, cmap=plt.cm.RdYlBu)

    test_labels = prep["y_test"].to_numpy()
    colors = np.where(test_labels == 1, "#E45756", "#4C78A8")
    markers = np.where(test_labels == 1, "^", "o")

    for label, marker in [(0, "o"), (1, "^")]:
        mask = test_labels == label
        ax.scatter(
            prep["X_test_pca2"][mask, 0],
            prep["X_test_pca2"][mask, 1],
            c="#4C78A8" if label == 0 else "#E45756",
            marker=marker,
            s=55,
            edgecolors="black",
            linewidth=0.3,
            label="Lower Risk" if label == 0 else "Higher Risk"
        )

    ax.set_title(f"{kernel.upper()} Kernel Decision Regions on 2D PCA View", fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig
