import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prepare_binary_target(df: pd.DataFrame):
    """
    Binary label for logistic regression and multinomial NB comparison.
    0 = lower depression risk
    1 = higher depression risk
    """
    y = (df["depression_score"] > 14).astype(int)
    return y


def get_regression_features(df: pd.DataFrame):
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


def prepare_regression_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X, feature_names = get_regression_features(df)
    y = prepare_binary_target(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Logistic Regression -> scaled
    scaler = StandardScaler()
    X_train_log = scaler.fit_transform(X_train_raw)
    X_test_log = scaler.transform(X_test_raw)

    # Multinomial NB -> non-negative
    X_train_nb = X_train_raw.copy()
    X_test_nb = X_test_raw.copy()

    for col in X_train_nb.columns:
        min_val = min(X_train_nb[col].min(), X_test_nb[col].min())
        if min_val < 0:
            X_train_nb[col] = X_train_nb[col] - min_val
            X_test_nb[col] = X_test_nb[col] - min_val

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_log": X_train_log,
        "X_test_log": X_test_log,
        "X_train_nb": X_train_nb,
        "X_test_nb": X_test_nb
    }


def run_logistic_and_nb(prep: dict):
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    log_model = LogisticRegression(max_iter=2000, random_state=42)
    log_model.fit(prep["X_train_log"], y_train)
    log_preds = log_model.predict(prep["X_test_log"])

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(prep["X_train_nb"], y_train)
    nb_preds = nb_model.predict(prep["X_test_nb"])

    results = {
        "Logistic Regression": {
            "model": log_model,
            "preds": log_preds,
            "accuracy": accuracy_score(y_test, log_preds),
            "confusion_matrix": confusion_matrix(y_test, log_preds),
            "report": classification_report(y_test, log_preds, output_dict=True, zero_division=0)
        },
        "Multinomial NB": {
            "model": nb_model,
            "preds": nb_preds,
            "accuracy": accuracy_score(y_test, nb_preds),
            "confusion_matrix": confusion_matrix(y_test, nb_preds),
            "report": classification_report(y_test, nb_preds, output_dict=True, zero_division=0)
        }
    }

    return results


def plot_confusion_matrix(cm, labels=("Low Risk", "High Risk"), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
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
            "Model": name,
            "Accuracy": round(out["accuracy"], 4)
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
