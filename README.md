# Dimensionality Reduction: Compress MNIST with Scikit-Learn

Chapter 7 of Hands-On Machine Learning as a practitioner applies it: fit PCA on MNIST, choose the number of components from the explained-variance curve, compress and reconstruct images, speed it up with randomized and incremental PCA, unroll a Swiss roll with LLE, map digits with t-SNE, then save a PCA-plus-classifier pipeline and serve it.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_mnist_subset
- [x] **2.** fit_pca
- [x] **3.** components_for_variance
- [x] **4.** explained_variance_curve
- [x] **5.** compress
- [x] **6.** reconstruction_error
- [x] **7.** compression_ratio
- [x] **8.** randomized_pca
- [x] **9.** incremental_pca
- [x] **10.** random_projection
- [x] **11.** unroll_swiss_roll
- [x] **12.** tsne_map
- [ ] **13.** classifier_on_compressed
- [ ] **14.** pca_classifier_pipeline
- [ ] **15.** save_and_reload_pipeline
- [ ] **16.** predict_images

---

Built on Deep-ML.
