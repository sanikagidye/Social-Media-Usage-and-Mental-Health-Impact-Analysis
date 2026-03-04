import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_FEATURES = [
    "daily_usage_hours", "late_night_hours", "comparison_content_pct",
    "fomo_score", "sessions_per_day", "engagement_ratio"
]


def ensure_depression_severity(df: pd.DataFrame):
    if "depression_severity" not in df.columns and "depression_score" in df.columns:
        df = df.copy()
        df["depression_severity"] = pd.cut(
            df["depression_score"],
            bins=[-1, 9, 14, 19, 27],
            labels=["Low", "Moderate", "Mod-Severe", "Severe"]
        )
    return df


def prep_clustering_data(df: pd.DataFrame, features=None, label_col="depression_severity"):
    df = ensure_depression_severity(df)

    if features is None:
        features = DEFAULT_FEATURES

    # labeled dataset sample (for "before")
    labeled_sample = df[features + [label_col]].head(10) if label_col in df.columns else df[features].head(10)

    # remove label and keep numeric only
    X = df[features].copy().fillna(0)

    labels = df[label_col].copy() if label_col in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # optional PCA to 3D (rubric allows / recommends)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    variance_retained = float(pca.explained_variance_ratio_.sum())

    return {
        "features": features,
        "labeled_sample": labeled_sample,
        "X_raw": X,
        "labels": labels,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "pca": pca,
        "X_pca": X_pca,
        "variance_retained": variance_retained,
    }


def silhouette_k_search(X_pca: np.ndarray, k_range=range(2, 11), random_state=42):
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        clusters = km.fit_predict(X_pca)
        scores.append(silhouette_score(X_pca, clusters))
    scores = np.array(scores)
    top_k = [list(k_range)[i] for i in np.argsort(scores)[-3:][::-1]]
    return scores, top_k


def plot_silhouette_curve(k_range, scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), list(scores), "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("K", fontweight="bold")
    ax.set_ylabel("Silhouette Score", fontweight="bold")
    ax.set_title("Silhouette Method", fontweight="bold")
    ax.grid(alpha=0.3)
    return fig


def run_kmeans(X_pca: np.ndarray, k: int, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = km.fit_predict(X_pca)
    centroids = km.cluster_centers_
    return clusters, centroids


def _label_colors(labels: pd.Series):
    """
    Converts categorical labels to numeric codes just for coloring plots (NOT for modeling).
    """
    codes, uniques = pd.factorize(labels.astype(str), sort=True)
    return codes, list(uniques)


def plot_kmeans_with_original_label_colors(X_pca: np.ndarray, centroids: np.ndarray, k: int, labels: pd.Series):
    """
    Rubric requirement: use color for original labels AND show kmeans centroids.
    We plot PC1 vs PC2 and PC1 vs PC2 vs PC3.
    """
    fig = plt.figure(figsize=(14, 6))
    codes, unique_names = _label_colors(labels)

    # 2D
    ax1 = fig.add_subplot(121)
    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=codes, alpha=0.65, s=45, cmap="tab10")
    ax1.scatter(
        centroids[:, 0], centroids[:, 1],
        c="red", marker="X", s=280, edgecolors="black", linewidth=2,
        label="Centroids", zorder=10
    )
    ax1.set_xlabel("PC1", fontweight="bold")
    ax1.set_ylabel("PC2", fontweight="bold")
    ax1.set_title(f"K-Means (K={k}) - Colored by ORIGINAL Labels (2D)", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    # create legend mapping label->color
    handles = []
    for i, name in enumerate(unique_names):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                 label=name, markerfacecolor=plt.cm.tab10(i/10),
                                 markersize=8))
    ax1.legend(handles=handles[:len(unique_names)], title="Original Label", loc="upper right")

    # 3D
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=codes, alpha=0.65, s=35, cmap="tab10")
    ax2.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c="red", marker="X", s=200, edgecolors="black", linewidth=2,
        label="Centroids", zorder=10
    )
    ax2.set_xlabel("PC1", fontweight="bold")
    ax2.set_ylabel("PC2", fontweight="bold")
    ax2.set_zlabel("PC3", fontweight="bold")
    ax2.set_title(f"K-Means (K={k}) - Colored by ORIGINAL Labels (3D)", fontweight="bold")

    return fig


def plot_dendrogram(X_pca: np.ndarray, method="ward", truncate_mode="lastp", p=30):
    linkage_matrix = linkage(X_pca, method=method)
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(linkage_matrix, ax=ax, truncate_mode=truncate_mode, p=p)
    ax.set_xlabel("Sample Index", fontweight="bold")
    ax.set_ylabel("Distance", fontweight="bold")
    ax.set_title(f"Dendrogram ({method.title()} Linkage)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    return fig


def run_dbscan(X_pca: np.ndarray, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(X_pca)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = int(np.sum(clusters == -1))
    return clusters, n_clusters, n_noise


def plot_dbscan_2d_3d(X_pca: np.ndarray, clusters: np.ndarray):
    fig = plt.figure(figsize=(14, 6))
    labels_set = sorted(set(clusters))

    ax1 = fig.add_subplot(121)
    for label in labels_set:
        mask = clusters == label
        if label == -1:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c="gray", marker="x", alpha=0.6, label="Noise")
        else:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, label=f"Cluster {label}")
    ax1.set_xlabel("PC1", fontweight="bold")
    ax1.set_ylabel("PC2", fontweight="bold")
    ax1.set_title("DBSCAN - 2D", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    for label in labels_set:
        mask = clusters == label
        if label == -1:
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], c="gray", marker="x", alpha=0.6, label="Noise")
        else:
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], alpha=0.6, label=f"Cluster {label}")
    ax2.set_xlabel("PC1", fontweight="bold")
    ax2.set_ylabel("PC2", fontweight="bold")
    ax2.set_zlabel("PC3", fontweight="bold")
    ax2.set_title("DBSCAN - 3D", fontweight="bold")
    ax2.legend()

    return fig