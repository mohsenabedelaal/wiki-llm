# Stochastic Gradient Boosting

**Summary**: Friedman's modification that fits each base learner on a random subsample of the training set (without replacement). Improves generalization and speeds training.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## The modification

At each iteration, draw a subsample of size $f \cdot n$ (without replacement, with $0 < f \leq 1$) and fit the next weak learner only on that subsample (source: Gradient boosting.md). $f = 1$ recovers deterministic gradient boosting; smaller $f$ injects randomness.

The motivation came from Breiman's bagging — but where bagging samples *with* replacement at the same size, this samples *without* replacement at a smaller size (source: Gradient boosting.md).

## Recommended values

Friedman observed that $0.5 \leq f \leq 0.8$ works well for small and moderate-sized training sets. $f = 0.5$ — using half the data per tree — is a common default (source: Gradient boosting.md).

## Why it helps

1. **[[regularization]] effect.** Each tree sees only a partial view of the data, so the ensemble averages over more diverse functions, reducing overfitting (source: Gradient boosting.md).
2. **Speed.** Smaller training sets per iteration → faster tree fits (source: Gradient boosting.md).
3. **Out-of-bag error.** Predictions on instances *not* used to build a given tree can serve as a validation estimate without needing a separate hold-out. But OOB tends to underestimate true improvement and the optimal number of iterations (source: Gradient boosting.md).

## In XGBoost

XGBoost exposes this as `subsample` (row subsampling) and `colsample_bytree` / `colsample_bylevel` / `colsample_bynode` (feature subsampling, an additional axis not in Friedman's original).

## Related pages

- [[regularization]]
- [[gradient-boosting]]
- [[xgboost]]
