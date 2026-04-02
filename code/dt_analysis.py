import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prepare_dt_target(df: pd.DataFrame):
    y = pd.cut(
        df["depression_score"],
        bins=[-1, 9, 14, 19, 27],
        labels=["Low", "Moderate", "Mod-Severe", "Severe"]
    )
    return y.astype(str)


def get_dt_features(df: pd.DataFrame):
    candidate_features = [
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
        "engagement_ratio"
    ]
    features = [f for f in candidate_features if f in df.columns]
    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    return X, features


def prepare_dt_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X, feature_names = get_dt_features(df)
    y = prepare_dt_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def _root_feature_name(tree_model, feature_names):
    idx = tree_model.tree_.feature[0]
    if idx == -2:
        return "Leaf"
    return feature_names[idx]


def build_three_different_trees(prep: dict):
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    feature_sets = []
    all_features = list(prep["feature_names"])

    # Tree 1: all features
    feature_sets.append(all_features)

    # Fit temp tree to identify root feature
    temp1 = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
    temp1.fit(X_train[all_features], y_train)
    root1 = _root_feature_name(temp1, all_features)

    # Tree 2: exclude root1
    features2 = [f for f in all_features if f != root1]
    if len(features2) < 2:
        features2 = all_features
    feature_sets.append(features2)

    temp2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    temp2.fit(X_train[features2], y_train)
    root2 = _root_feature_name(temp2, features2)

    # Tree 3: exclude root1 and root2
    features3 = [f for f in all_features if f not in [root1, root2]]
    if len(features3) < 2:
        features3 = all_features
    feature_sets.append(features3)

    configs = [
        ("Tree 1", DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42), feature_sets[0]),
        ("Tree 2", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42), feature_sets[1]),
        ("Tree 3", DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=20, random_state=42), feature_sets[2]),
    ]

    results = {}

    for name, model, feat_list in configs:
        model.fit(X_train[feat_list], y_train)
        preds = model.predict(X_test[feat_list])

        results[name] = {
            "model": model,
            "features_used": feat_list,
            "root_feature": _root_feature_name(model, feat_list),
            "preds": preds,
            "accuracy": accuracy_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=sorted(y_test.unique())),
            "labels": sorted(y_test.unique()),
            "report": classification_report(y_test, preds, output_dict=True, zero_division=0)
        }

    return results


def plot_decision_tree_model(model, feature_names, class_names, title="Decision Tree"):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def accuracy_table(results: dict):
    rows = []
    for name, out in results.items():
        rows.append({
            "Tree": name,
            "Root Feature": out["root_feature"],
            "Accuracy": round(out["accuracy"], 4)
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
