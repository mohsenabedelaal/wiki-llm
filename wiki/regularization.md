# Regularization (in Gradient Boosting)

**Summary**: Techniques that constrain the fit to prevent overfitting. Gradient boosting fits training data aggressively, so regularization is essential.

**Sources**: `Gradient boosting.md`, `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Why it matters

Fitting the training set too closely degrades [[gradient-boosting]]'s generalization. The [[bias-variance-tradeoff]] is real and acute for boosted ensembles: keep adding trees and training error keeps falling, but validation error eventually rises (source: Gradient boosting.md).

The XGBoost tutorial makes this structural: the [[objective-function]] is *defined* as training loss plus a regularization term, not just a training loss with regularization tacked on (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Techniques

**Iteration count $M$** — More trees lower training error but raise overfitting risk. Pick $M$ by monitoring validation error (source: Gradient boosting.md).

**Tree depth / [[tree-size]]** — Deeper trees express higher-order feature interactions but overfit more readily. Typical $J \in [4, 8]$ leaves (source: Gradient boosting.md).

**[[shrinkage]]** — A learning-rate multiplier $\nu < 1$ on each tree's contribution. Small $\nu$ (e.g., $< 0.1$) dramatically improves generalization at the cost of needing more iterations (source: Gradient boosting.md).

**[[stochastic-gradient-boosting]]** — At each iteration, fit on a random subsample of size $f \cdot n$ (typically $f \in [0.5, 0.8]$). Reduces overfitting and speeds training (source: Gradient boosting.md).

**Minimum observations per leaf** — Skip splits that would produce leaves with too few instances. Reduces variance in leaf predictions (source: Gradient boosting.md).

**[[complexity-penalty]]** — Penalize the number of leaves $T$ via a term $\gamma T$. Equivalent to post-pruning branches whose gain doesn't exceed $\gamma$. Used in [[xgboost]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

**[[l2-leaf-penalty]]** — Penalize the squared leaf weights via $\tfrac{1}{2}\lambda \sum w_j^2$. Smooths leaf values and shrinks them toward zero (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## In practice

XGBoost users typically tune some combination of: `eta` (shrinkage), `max_depth` or `max_leaves` (tree size), `subsample` and `colsample_*` (stochasticity), `min_child_weight` (min-observations / min-Hessian), `gamma` (complexity penalty), and `reg_lambda` (L2 leaf penalty). All map directly to the techniques above.

## Related pages

- [[gradient-boosting]]
- [[xgboost]]
- [[shrinkage]]
- [[stochastic-gradient-boosting]]
- [[tree-size]]
- [[complexity-penalty]]
- [[l2-leaf-penalty]]
- [[bias-variance-tradeoff]]
- [[objective-function]]
