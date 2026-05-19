# Decision Tree Ensembles

**Summary**: A model that sums predictions from many [[cart]] trees. Both random forests and gradient-boosted trees use this same model — the difference is in how they are trained.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## The model

For $K$ trees $f_k$ drawn from the space $\mathcal{F}$ of all CARTs, the ensemble predicts

$$\hat{y}_i = \sum_{k=1}^K f_k(x_i), \quad f_k \in \mathcal{F}.$$

A single tree is rarely strong enough in practice; ensembles complement each other, with each tree correcting where others are weak (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Same model, different training

A subtle but important point from the XGBoost docs: random forests and gradient-boosted trees are the *same model* — tree ensembles. They differ only in how trees are trained (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md):

- **Random forest**: trees trained in parallel on bootstrap samples (bagging) with random feature subsets.
- **Gradient-boosted trees**: trees trained sequentially, each fit to the [[pseudo-residuals]] of the current ensemble via [[additive-training]].

This implies a prediction service for tree ensembles works for both (see Treelite).

## Objective

The training objective for the ensemble is:

$$\text{obj}(\theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \omega(f_k)$$

where $\omega(f_k)$ measures tree complexity (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). See [[objective-function]] and [[complexity-penalty]].

## Related pages

- [[cart]]
- [[gradient-boosting]]
- [[xgboost]]
- [[additive-training]]
- [[objective-function]]
- [[boosting]]
