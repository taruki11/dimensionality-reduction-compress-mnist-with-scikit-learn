"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Dimensionality reduction with scikit-learn (Hands-On ML, chapter 7).

Story: PCA on MNIST, the explained-variance curve and the 95% rule, compression and
reconstruction error, the randomized and incremental solvers, a random projection,
LLE on a Swiss roll and t-SNE on digits, a classifier that keeps its accuracy on
a fifth of the inputs, and a saved PCA-plus-classifier pipeline serving raw images.
"""
import os
import tempfile
import numpy as np


def main() -> None:
    X, y = load_mnist_subset(3000)
    pca = fit_pca(X)
    d95 = components_for_variance(pca, 0.95)
    curve = explained_variance_curve(pca, [1, 10, 50, d95, 300, 784])
    print(f"MNIST slice {X.shape}; cumulative explained variance: " + "  ".join(f"{k}:{v:.3f}" for k, v in curve.items()))
    print(f"95% of the variance needs {d95} of 784 components")

    # ---- compress / reconstruct ----
    for k in (10, 50, d95):
        p, Xr = compress(X, k)
        print(f"  {k:>3} components: {compression_ratio(X, Xr):>5.2f}x smaller, reconstruction MSE {reconstruction_error(p, X):,.1f} per pixel")

    # ---- faster and cheaper ----
    r = randomized_pca(X, d95)
    print(f"randomized solver: explained-variance gap vs exact {r['variance_gap']:.5f}")
    ipca = incremental_pca(X, d95, n_batches=10)
    print(f"incremental PCA in 10 batches: reconstruction MSE {reconstruction_error(ipca, X):,.1f}")
    rp = random_projection(X, 300)
    print(f"random projection to 300 dims: mean distance ratio {rp['mean_ratio']:.3f}, max distortion {rp['max_distortion']:.3f}")

    # ---- manifolds and maps ----
    sw = unroll_swiss_roll()
    print(f"\nLLE unrolls the Swiss roll: |corr(t, embedding)| = {sw['t_correlation']:.3f}")
    ts = tsne_map(X, y, n=1000)
    print(f"t-SNE map of 1000 digits: between/within class separation {ts['separation']:.2f}")

    # ---- the payoff ----
    cc = classifier_on_compressed(X, y, d95)
    print(f"\nlogistic regression: raw {cc['n_features'][0]} pixels -> {cc['raw_accuracy']:.3f}; "
          f"PCA {cc['n_features'][1]} features -> {cc['pca_accuracy']:.3f}")

    # ---- ship ----
    pipe = pca_classifier_pipeline(variance=0.95).fit(X, y)
    path = os.path.join(tempfile.gettempdir(), "pca_classifier.pkl")
    served = save_and_reload_pipeline(pipe, path)
    with np.load(os.path.join(tempfile.gettempdir(), "mnist.npz")) as z:
        imgs, truth = z["x_test"][:6], z["y_test"][:6]
    print(f"served pipeline ({served.steps[0][1].n_components_} components) on 6 raw images: {predict_images(served, imgs)} (truth {truth.tolist()})")


if __name__ == "__main__":
    main()

