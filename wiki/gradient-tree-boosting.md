# Gradient Tree Boosting (TreeBoost)

**Summary**: The specialization of [[gradient-boosting]] where each weak learner is a fixed-size [[cart]]. Friedman's "TreeBoost" refinement assigns a separate optimal step size to each leaf region rather than one per tree.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Setup

A tree $h_m$ with $J_m$ leaves partitions input space into disjoint regions $R_{1m}, \dots, R_{J_m m}$ and predicts a constant $b_{jm}$ in each:

$$h_m(x) = \sum_{j=1}^{J_m} b_{jm} \mathbf{1}_{R_{jm}}(x).$$

The generic [[gradient-boosting]] update multiplies the entire tree by one shared $\gamma_m$ chosen by line search:

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x).$$

(source: Gradient boosting.md)

## TreeBoost: per-region $\gamma$

Friedman's refinement picks a *separate* optimal $\gamma_{jm}$ for each leaf region instead of a single $\gamma_m$ for the whole tree:

$$F_m(x) = F_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} \mathbf{1}_{R_{jm}}(x), \quad \gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

(source: Gradient boosting.md). The tree-fitting coefficients $b_{jm}$ are simply discarded — only the partition structure is reused. For squared-error loss, $\gamma_{jm}$ coincides with $b_{jm}$ (the regional mean), but for other losses TreeBoost produces meaningfully different (and better) leaf values.

## Why per-region is better

Different leaves cover different parts of input space with different residual distributions. A single per-tree $\gamma$ is a compromise; per-region $\gamma_{jm}$ does the right thing locally and improves fit (source: Gradient boosting.md).

## Connection to XGBoost

[[xgboost]] takes this even further: rather than first fitting a tree to pseudo-residuals and then refitting leaf values, it derives the optimal leaf weights $w_j^* = -G_j/(H_j + \lambda)$ directly from the regularized [[structure-score]] using gradients *and* Hessians (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[gradient-boosting]]
- [[cart]]
- [[xgboost]]
- [[structure-score]]
- [[tree-size]]
- [[weak-learner]]
