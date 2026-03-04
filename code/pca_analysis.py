import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_numeric_features(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if "user_id" not in c.lower()]
    return numeric_features


def prepare_pca_data(df: pd.DataFrame):
    numeric_features = get_numeric_features(df)
    X = df[numeric_features].copy().fillna(0)
    return X, numeric_features


def scale_data(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def run_pca(X_scaled: np.ndarray):
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)

    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)
    eigenvalues = pca_full.explained_variance_

    return {
        "pca_2d": pca_2d,
        "X_pca_2d": X_pca_2d,
        "pca_3d": pca_3d,
        "X_pca_3d": X_pca_3d,
        "pca_full": pca_full,
        "cumulative_variance": cumulative_variance,
        "n_components_95": n_components_95,
        "eigenvalues": eigenvalues,
    }


def pca_loadings_table(pca_model: PCA, feature_names, top_n=10):
    """
    Loadings = component weights. High abs weight => more influence on that PC.
    Returns a tidy table showing top contributing features for PC1/PC2/PC3.
    """
    comps = pca_model.components_
    n_comp = comps.shape[0]
    feature_names = list(feature_names)

    tables = []
    for i in range(n_comp):
        weights = comps[i]
        dfw = pd.DataFrame({
            "feature": feature_names,
            "loading": weights,
            "abs_loading": np.abs(weights),
        }).sort_values("abs_loading", ascending=False).head(top_n)

        dfw.insert(0, "PC", f"PC{i+1}")
        tables.append(dfw[["PC", "feature", "loading", "abs_loading"]])

    return pd.concat(tables, ignore_index=True)


def plot_pca_2d(X_pca_2d: np.ndarray, pca_2d: PCA, color_values=None, color_label="Color"):
    fig, ax = plt.subplots(figsize=(10, 6))
    if color_values is None:
        ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6, s=50)
    else:
        sc = ax.scatter(
            X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=color_values, alpha=0.6, s=50, cmap="RdYlGn_r"
        )
        plt.colorbar(sc, ax=ax, label=color_label)

    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold")
    ax.set_title("PCA: 2D Projection", fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    return fig


def plot_pca_3d(X_pca_3d: np.ndarray, pca_3d: PCA, color_values=None, color_label="Color"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_values is None:
        ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.6, s=50)
    else:
        sc = ax.scatter(
            X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
            c=color_values, alpha=0.6, s=50, cmap="RdYlGn_r"
        )
        plt.colorbar(sc, ax=ax, label=color_label, shrink=0.5)

    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold")
    ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)", fontweight="bold")
    ax.set_title("PCA: 3D Projection", fontweight="bold")
    return fig


def plot_cumulative_variance(cumulative_variance: np.ndarray, n_components_95: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-", linewidth=2)
    ax.axhline(y=0.95, color="r", linestyle="--", linewidth=2, label="95% Threshold")
    ax.axvline(x=n_components_95, color="g", linestyle="--", linewidth=2, label=f"{n_components_95} Components")
    ax.set_xlabel("Number of Components", fontweight="bold")
    ax.set_ylabel("Cumulative Variance", fontweight="bold")
    ax.set_title("Cumulative Variance Explained", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def eigenvalue_table(eigenvalues: np.ndarray, explained_variance_ratio: np.ndarray, top_n: int = 10):
    n = min(top_n, len(eigenvalues))
    return pd.DataFrame({
        "Component": range(1, n + 1),
        "Eigenvalue": eigenvalues[:n],
        "Variance %": explained_variance_ratio[:n] * 100
    })