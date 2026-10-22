"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist_subset
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 2 - fit_pca
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 3 - components_for_variance
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 4 - explained_variance_curve
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 5 - compress
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 6 - reconstruction_error
import os
import tempfile
import urllib.request

import joblib
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# ── Part 1：PCA 和解释方差 ────────────────────────────────────

# Step 1
def load_mnist_subset(n=3000):
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")

    # 文件已经缓存时就不重复下载
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(-1, 784).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y


# Step 2
def fit_pca(X, n_components=None, random_state=42):
    model = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def components_for_variance(pca, threshold=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # 找到累计解释方差第一次达到 threshold 的位置
    indices = np.flatnonzero(cumulative + 1e-12 >= threshold)

    if len(indices) == 0:
        return int(len(cumulative))

    return int(indices[0] + 1)


# Step 4
def explained_variance_curve(pca, ks):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return {
        k: round(float(cumulative[k - 1]), 4)
        for k in ks
    }


# ── Part 2：压缩、重构、误差 ──────────────────────────────────

# Step 5
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = pca.transform(X)
    return pca, X_reduced


# Step 6
def reconstruction_error(pca, X):
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    return float(mse)


# Step 7
def compression_ratio(X, X_reduced):
    ratio = X.shape[1] / X_reduced.shape[1]
    return round(float(ratio), 2)


# ── Part 3：更快的 PCA 与随机投影 ─────────────────────────────

# Step 8
def randomized_pca(X, n_components, random_state=42):
    randomized = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X)

    exact, _ = compress(
        X,
        n_components=n_components,
        random_state=random_state,
    )

    variance_gap = abs(
        float(randomized.explained_variance_ratio_.sum())
        - float(exact.explained_variance_ratio_.sum())
    )

    return {
        "pca": randomized,
        "variance_gap": float(variance_gap),
    }


# Step 9
def incremental_pca(X, n_components, n_batches=10):
    model = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        if len(batch) > 0:
            model.partial_fit(batch)

    return model


# Step 10
def random_projection(X, n_components, random_state=42, n_pairs=500):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), size=(n_pairs, 2))

    i = pairs[:, 0]
    j = pairs[:, 1]

    original_distances = np.linalg.norm(X[i] - X[j], axis=1)
    projected_distances = np.linalg.norm(
        X_reduced[i] - X_reduced[j],
        axis=1,
    )

    # 忽略同一个样本和原始距离为零的样本对
    valid = (i != j) & (original_distances > 0)

    ratios = projected_distances[valid] / original_distances[valid]

    if len(ratios) == 0:
        mean_ratio = 1.0
        max_distortion = 0.0
    else:
        mean_ratio = float(np.mean(ratios))
        max_distortion = float(np.max(np.abs(ratios - 1.0)))

    return {
        "X_reduced": X_reduced,
        "mean_ratio": mean_ratio,
        "max_distortion": max_distortion,
    }


# ── Part 4：流形降维、t-SNE 和分类器 ──────────────────────────

# Step 11
def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state,
    )

    model = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state,
    )
    X_2d = model.fit_transform(X)

    correlations = [
        abs(float(np.corrcoef(t, X_2d[:, j])[0, 1]))
        for j in range(X_2d.shape[1])
    ]

    return {
        "X_2d": X_2d,
        "t_correlation": float(max(correlations)),
    }


# Step 12
def tsne_map(X, y, n=1000, random_state=42):
    X_sub = np.asarray(X)[:n]
    y_sub = np.asarray(y)[:n]

    model = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state,
    )
    X_2d = model.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = []
    within_class_distances = []

    for label in classes:
        points = X_2d[y_sub == label]
        centroid = points.mean(axis=0)

        centroids.append(centroid)
        within_class_distances.append(
            float(np.linalg.norm(points - centroid, axis=1).mean())
        )

    # 所有不同类别质心之间的平均距离
    between_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(
                float(np.linalg.norm(centroids[i] - centroids[j]))
            )

    mean_between = float(np.mean(between_distances))
    mean_within = float(np.mean(within_class_distances))

    separation = mean_between / mean_within

    return {
        "X_2d": X_2d,
        "separation": float(separation),
    }


# Step 13
def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 基线：直接用原始像素训练
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(X_train, y_train)
    raw_accuracy = float(
        accuracy_score(y_test, raw_model.predict(X_test))
    )

    # PCA 只在训练集上拟合，避免使用测试集信息
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_model = LogisticRegression(max_iter=2000)
    pca_model.fit(X_train_pca, y_train)
    pca_accuracy = float(
        accuracy_score(y_test, pca_model.predict(X_test_pca))
    )

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (int(X.shape[1]), int(X_train_pca.shape[1])),
    }


# ── Part 5：组合流水线、保存、预测图片 ─────────────────────────

# Step 14
def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state,
        ),
        LogisticRegression(max_iter=2000),
    )


# Step 15
def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)


# Step 16
def predict_images(pipeline, images):
    images = np.asarray(images, dtype=np.float64)
    X = images.reshape(len(images), 784)

    predictions = pipeline.predict(X)
    return predictions.astype(int).tolist()

# Step 7 - compression_ratio (not yet solved)
# TODO: implement

# Step 8 - randomized_pca (not yet solved)
# TODO: implement

# Step 9 - incremental_pca (not yet solved)
# TODO: implement

# Step 10 - random_projection (not yet solved)
# TODO: implement

# Step 11 - unroll_swiss_roll (not yet solved)
# TODO: implement

# Step 12 - tsne_map (not yet solved)
# TODO: implement

# Step 13 - classifier_on_compressed (not yet solved)
# TODO: implement

# Step 14 - pca_classifier_pipeline (not yet solved)
# TODO: implement

# Step 15 - save_and_reload_pipeline (not yet solved)
# TODO: implement

# Step 16 - predict_images (not yet solved)
# TODO: implement

