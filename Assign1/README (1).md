# Major Assignment 1: Hamming PUF
**CS771: Introduction to Machine Learning — IIT Kanpur**

| Name | Roll Number | Email |
|------|-------------|-------|
| Kushagra Nayak | 240586 | kushagran24@iitk.ac.in |
| Prathu Agarwal | 240782 | prathuag24@iitk.ac.in |
| Abhishek Sardar | 240041 | abhishekh24@iitk.ac.in |
| Pankaj Khoriya | 240729 | kpankaj24@iitk.ac.in |

---

## Overview

This project implements a machine learning attack on a **Hamming Physical Unclonable Function (PUF)**. Given a set of Challenge-Response Pairs (CRPs), we learn a linear classifier that predicts the PUF response to any new challenge, without knowing the secret bits or threshold directly.

The core insight is that the PUF response can be expressed as a linear function over a 288-dimensional feature map — enabling an SVM to learn the secret from CRP data alone.

---

## Files

| File | Description |
|------|-------------|
| `submit.py` | Main submission file with `my_map()` and `my_params()` |
| `Report.pdf` | Full report with derivations, experiments, and analysis |
| `README.md` | This file |

---

## Feature Map (`my_map`)

Challenges `c ∈ {0,1}^32` are transformed into a 288-dimensional feature vector:

1. **Bit conversion:** Each bit `c_i` is mapped to `x_i = 1 − 2c_i ∈ {−1, +1}`
2. **Linear terms (32):** `x_0, x_1, ..., x_31`
3. **Cross-product terms (256):** `x_i * x_j` for all even `i`, odd `j` (16 × 16 pairs)

This map is derived from the mathematical structure of `h_e · h_o` (the product of even- and odd-indexed Hamming weights), which governs the PUF response.

**Minimum feature dimensionality:** D = 32 + 256 = **288** (linear terms cannot be replaced by cross-products).

---

## Model & Hyperparameters (`my_params`)

The model uses **`LinearSVC`** (scikit-learn) with the following parameters, chosen based on systematic experiments:

```python
{
    'C'       : 0.9,
    'tol'     : 1e-4,
    'max_iter': 5000,
    'dual'    : False,   # primal solver — faster when samples >> features
}
```

### Hyperparameter Findings

**Loss function:** `squared_hinge` with `dual=False` is ~9× faster than `hinge` with `dual=True`, with identical test accuracy (99.80%).

**Regularisation C:** Both LinearSVC and LogisticRegression improve sharply up to C ≈ 0.1 then plateau. C = 0.9 is the sweet spot — full training accuracy with no overfitting or excess training time.

**Tolerance:** `tol = 1e-4` is a safe default; looser tolerances hurt LogisticRegression convergence while tighter ones add training time with no accuracy gain.

**Penalty (ℓ1 vs ℓ2):** ℓ1 gives marginally better accuracy for LinearSVC but is ~24× slower. Since training time is penalised in grading, **ℓ2 is used**.

---

## Training Set Size Analysis

Test accuracy as a function of training set size (10 trials each, fixed 2500-CRP test set):

| Accuracy Threshold | Approx. CRPs Needed |
|--------------------|---------------------|
| ≥ 95% | ~2,500 |
| ≥ 97% | ~3,000 |
| ≥ 99% | ~4,000 |

Accuracy rises steeply between 500–3500 training points, then flattens — consistent with the mapped data being nearly linearly separable.

---

## Requirements

- Python 3.x
- `numpy`
- `scikit-learn`

No other libraries (scipy, keras, tensorflow, etc.) are used or required.

---

## Usage

The evaluation script will call `my_map` and `my_params` directly:

```python
from submit import my_map, my_params

X_map = my_map(X_raw)          # Transform challenges → 288-d features
params = my_params(X_map, X_raw, y)  # Get hyperparameter dict for LinearSVC
```

---

## References

1. Scikit-learn: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation.
2. Purushottam Kar. *CS771: Introduction to Machine Learning*, Course Materials. IIT Kanpur, 2025–26.
