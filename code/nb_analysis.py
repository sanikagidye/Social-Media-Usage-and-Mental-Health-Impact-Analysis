import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prepare_nb_target(df: pd.DataFrame):
    """
    Creates a multiclass depression severity label for supervised NB classification.
    """
    y = pd.cut(
        df["depression_score"],
        bins=[-1, 9, 14, 19, 27],
        labels=["Low", "Moderate", "Mod-Severe", "Severe"]
    )
    return y.astype(str)


def get_base_features(df: pd.DataFrame):
    """
    Feature set chosen to be relevant, mostly numeric, and safe for supervised models.
    """
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


def prepare_nb_datasets(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Prepares three datasets:
    1) Multinomial NB -> non-negative numeric data
    2) Gaussian NB -> scaled continuous numeric data
    3) Bernoulli NB -> binary transformed data
    """
    X, feature_names = get_base_features(df)
    y = prepare_nb_target(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Multinomial NB: must be non-negative
    X_train_multi = X_train_raw.copy()
    X_test_multi = X_test_raw.copy()

    for col in X_train_multi.columns:
        min_val = min(X_train_multi[col].min(), X_test_multi[col].min())
        if min_val < 0:
            X_train_multi[col] = X_train_multi[col] - min_val
            X_test_multi[col] = X_test_multi[col] - min_val

    # Gaussian NB: scaled continuous
    scaler = StandardScaler()
    X_train_gauss = scaler.fit_transform(X_train_raw)
    X_test_gauss = scaler.transform(X_test_raw)

    # Bernoulli NB: binary thresholded
    medians = X_train_raw.median()
    X_train_bern = (X_train_raw > medians).astype(int)
    X_test_bern = (X_test_raw > medians).astype(int)

    return {
        "feature_names": feature_names,
        "X_raw": X,
        "y": y,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_multi": X_train_multi,
        "X_test_multi": X_test_multi,
        "X_train_gauss": X_train_gauss,
        "X_test_gauss": X_test_gauss,
        "X_train_bern": X_train_bern,
        "X_test_bern": X_test_bern,
    }


def run_all_nb_models(prep: dict):
    models = {
        "Multinomial NB": MultinomialNB(alpha=1.0),
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB(alpha=1.0)
    }

    datasets = {
        "Multinomial NB": (prep["X_train_multi"], prep["X_test_multi"]),
        "Gaussian NB": (prep["X_train_gauss"], prep["X_test_gauss"]),
        "Bernoulli NB": (prep["X_train_bern"], prep["X_test_bern"])
    }

    y_train = prep["y_train"]
    y_test = prep["y_test"]

    results = {}

    for name, model in models.items():
        Xtr, Xte = datasets[name]
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        results[name] = {
            "model": model,
            "preds": preds,
            "accuracy": accuracy_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=sorted(y_test.unique())),
            "labels": sorted(y_test.unique()),
            "report": classification_report(y_test, preds, output_dict=True, zero_division=0)
        }

    return results


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
            "Model": name,
            "Accuracy": round(out["accuracy"], 4)
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
