# Tree Size

**Summary**: The number of terminal nodes (leaves) $J$ in each base tree. Controls the maximum order of feature interactions the model can capture.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Interpretation

$J$ is the *interaction depth*. A tree with $J$ leaves makes $J - 1$ splits, so it can use at most $J - 1$ features along any path (source: Gradient boosting.md):

- $J = 2$ — **decision stump**. Single split. Each tree captures one feature's main effect; no interactions.
- $J = 3$ — pairs of features can interact, but no higher-order interactions.
- Larger $J$ — proportionally higher-order interactions allowed.

## Practical guidance

Hastie, Tibshirani, and Friedman (in *Elements of Statistical Learning*) recommend $4 \leq J \leq 8$ for typical use, and report that results are fairly insensitive to $J$ within this range (source: Gradient boosting.md).

- $J = 2$ is **insufficient** for many applications.
- $J > 10$ is **rarely needed**.

## Connection to other regularizers

Tree size is one knob; depth, [[shrinkage]], and [[stochastic-gradient-boosting]] are others. In XGBoost, tree size is controlled either implicitly via `max_depth` or directly via `max_leaves`. The [[complexity-penalty]] $\gamma T$ also pushes toward smaller trees by making each additional leaf cost something in the [[objective-function]].

## Related pages

- [[regularization]]
- [[cart]]
- [[weak-learner]]
- [[gradient-tree-boosting]]
- [[complexity-penalty]]
