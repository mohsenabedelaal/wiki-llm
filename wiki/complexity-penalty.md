# Complexity Penalty ($\gamma T$)

**Summary**: A per-leaf cost added to the XGBoost objective. Discourages overly complex trees; in [[split-finding]], any candidate split with gain below $\gamma$ is rejected.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`, `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Definition

The tree complexity term in [[xgboost]]'s regularizer is:

$$\omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

The first part — $\gamma T$ — charges a flat cost $\gamma$ per leaf $T$ (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). The second part is the [[l2-leaf-penalty]].

## Effect: principled pruning

Because $\gamma T$ contributes additively to the [[structure-score]] and hence to the gain of every candidate split, the [[split-finding]] decision rule becomes: **split only when the gain exceeds $\gamma$**.

This is the same effect as classical tree pruning, derived directly from the regularized [[objective-function]] rather than imposed as a heuristic. The XGBoost tutorial frames this as a direct consequence of taking the principle "training loss + regularization" seriously (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Generic gradient boosting analogue

The Wikipedia article describes the same idea in generic [[gradient-tree-boosting]]: penalize model complexity by the proportional number of leaves, which corresponds to a post-pruning algorithm removing branches that fail to reduce loss by a threshold (source: Gradient boosting.md).

## XGBoost parameter

The `gamma` parameter in XGBoost (sometimes called `min_split_loss`) is exactly this $\gamma$.

## Related pages

- [[xgboost]]
- [[l2-leaf-penalty]]
- [[regularization]]
- [[split-finding]]
- [[structure-score]]
- [[objective-function]]
