# Additive Training

**Summary**: The strategy of growing a tree ensemble one tree at a time: at step $t$, fix all previously learned trees and choose the next tree $f_t$ to optimize the objective.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Motivation

The parameters of a tree ensemble are the *functions* $f_k$ themselves — each containing both a structure and leaf scores. Learning all $K$ trees jointly is intractable: tree structure isn't differentiable, so ordinary optimization fails (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). Additive training sidesteps this by being greedy: lock in everything learned so far, and pick the next tree to minimize the current objective.

## Stagewise update

At step $t$, the prediction is built as

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i), \quad \hat{y}_i^{(0)} = 0.$$

The objective at step $t$ becomes

$$\text{obj}^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \omega(f_t) + \text{constant}$$

(source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## From here to XGBoost's solver

To make this tractable for arbitrary losses, [[xgboost]] applies a [[second-order-taylor-approximation]] around $\hat{y}_i^{(t-1)}$, reducing the per-step objective to a function of the gradient $g_i$ and Hessian $h_i$ alone. This is the entry point to deriving the [[structure-score]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Relation to gradient boosting

Generic [[gradient-boosting]] is the same idea with a different per-step rule: fit a learner to the [[pseudo-residuals]] (gradient only) and do a line search. XGBoost's additive training is the second-order version of the same stagewise procedure.

## Related pages

- [[xgboost]]
- [[gradient-boosting]]
- [[second-order-taylor-approximation]]
- [[structure-score]]
- [[objective-function]]
