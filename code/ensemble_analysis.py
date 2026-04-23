import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ENSEMBLE_FEATURES = [
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


def get_ensemble_features(df: pd.DataFrame):
    features = [feature for feature in ENSEMBLE_FEATURES if feature in df.columns]
    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    return X, features


def prepare_ensemble_target(df: pd.DataFrame):
    return (df["depression_score"] > 14).astype(int)


def prepare_ensemble_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X, feature_names = get_ensemble_features(df)
    y = prepare_ensemble_target(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_ensemble_models(random_state=42):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state
        ),
        "Voting Ensemble": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
                ("et", ExtraTreesClassifier(n_estimators=200, random_state=random_state)),
                ("lr", Pipeline([
                    ("scaler", StandardScaler()),
                    ("log", LogisticRegression(max_iter=2000, random_state=random_state))
                ]))
            ],
            voting="hard"
        )
    }


def run_ensemble_models(prep: dict):
    results = {}
    labels = [0, 1]

    for name, model in build_ensemble_models().items():
        model.fit(prep["X_train_raw"], prep["y_train"])
        preds = model.predict(prep["X_test_raw"])
        results[name] = {
            "model": model,
            "preds": preds,
            "accuracy": accuracy_score(prep["y_test"], preds),
            "confusion_matrix": confusion_matrix(prep["y_test"], preds, labels=labels),
            "labels": labels,
        }

    return results


def ensemble_accuracy_table(results: dict):
    rows = []
    for name, details in results.items():
        rows.append({
            "Ensemble Method": name,
            "Accuracy": round(details["accuracy"], 4)
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(cm, labels=("Lower Risk", "Higher Risk"), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(cm, aspect="auto", cmap="Greens")
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


def plot_accuracy_bar(results: dict):
    ranking = ensemble_accuracy_table(results)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(
        ranking["Ensemble Method"],
        ranking["Accuracy"],
        color=["#4C78A8", "#54A24B", "#F58518", "#E45756"][:len(ranking)]
    )
    ax.set_ylim(0, max(0.7, ranking["Accuracy"].max() + 0.05))
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Ensemble Method Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)

    for bar, value in zip(bars, ranking["Accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()
    return fig


def plot_ensemble_concept():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")

    nodes = {
        "Input Data": (0.10, 0.52),
        "Model 1": (0.38, 0.78),
        "Model 2": (0.38, 0.52),
        "Model 3": (0.38, 0.26),
        "Combined Vote": (0.70, 0.52),
        "Final Prediction": (0.91, 0.52),
    }

    for label, (x_pos, y_pos) in nodes.items():
        width = 0.18 if label != "Final Prediction" else 0.16
        ax.add_patch(plt.Rectangle((x_pos - width / 2, y_pos - 0.10), width, 0.20, color="#D9E6F2", ec="#4C78A8", lw=2))
        ax.text(x_pos, y_pos, label, ha="center", va="center", fontweight="bold")

    arrows = [
        ("Input Data", "Model 1"),
        ("Input Data", "Model 2"),
        ("Input Data", "Model 3"),
        ("Model 1", "Combined Vote"),
        ("Model 2", "Combined Vote"),
        ("Model 3", "Combined Vote"),
        ("Combined Vote", "Final Prediction"),
    ]

    for start, end in arrows:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        ax.annotate("", xy=(x1 - 0.1, y1), xytext=(x0 + 0.1, y0), arrowprops=dict(arrowstyle="->", lw=2, color="#4C78A8"))

    ax.set_title("How Ensemble Learning Combines Multiple Models", fontweight="bold")
    return fig


def best_tree_ensemble(results: dict):
    eligible = {
        name: details for name, details in results.items()
        if hasattr(details["model"], "feature_importances_")
    }
    return max(eligible.items(), key=lambda item: item[1]["accuracy"])


def plot_feature_importance(model, feature_names, top_n=10, title="Feature Importance"):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#72B7B2")
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig
