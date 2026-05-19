# L2 Leaf Penalty ($\lambda$)

**Summary**: A quadratic penalty on each tree's leaf scores, $\tfrac{1}{2}\lambda \sum w_j^2$. Smooths leaf predictions toward zero and is what makes XGBoost's per-step objective strongly convex.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`, `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Definition

For a tree with leaf weights $w = (w_1, \dots, w_T)$, the L2 leaf penalty is

$$\tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

This appears alongside the [[complexity-penalty]] in the tree regularizer $\omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum w_j^2$ (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). The Wikipedia article notes the same kind of $\ell_2$ penalty on leaf values is a known regularizer for gradient boosted models (source: Gradient boosting.md).

## What it does

1. **Makes leaf weights closed-form solvable.** After the [[second-order-taylor-approximation]], the per-step objective is quadratic in each $w_j$, and the optimum is

   $$w_j^* = -\frac{G_j}{H_j + \lambda}.$$

   Without $\lambda$, leaves with small Hessian sums ($H_j \to 0$) could produce wildly large weights; $\lambda$ acts as a numerical stabilizer in the denominator (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

2. **Shrinks leaf values toward zero.** Larger $\lambda$ moves $w_j^*$ closer to zero, reducing each tree's contribution to the ensemble's prediction. This is analogous to ridge regression on the leaf outputs.

3. **Enters the [[structure-score]].** The same $\lambda$ appears in $\text{obj}^* = -\tfrac{1}{2}\sum_j G_j^2/(H_j + \lambda) + \gamma T$, so it directly influences which tree structures look favorable during [[split-finding]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## XGBoost parameter

In XGBoost this is `reg_lambda` (also called `lambda`). There is also an `reg_alpha` for L1 regularization on leaves, not covered by either source.

## Related pages

- [[xgboost]]
- [[complexity-penalty]]
- [[regularization]]
- [[second-order-taylor-approximation]]
- [[structure-score]]
- [[objective-function]]
