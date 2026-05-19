# XGBoost

**Summary**: "Extreme Gradient Boosting" — a scalable, regularized implementation of [[gradient-tree-boosting]] that formalizes the regularization term in the objective and uses a [[second-order-taylor-approximation]] of the loss.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`, `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## What it is

XGBoost is a library implementing [[gradient-boosting]] over [[cart]] tree ensembles. The name comes from Friedman's *Greedy Function Approximation: A Gradient Boosting Machine* (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). It is designed for scalability, portability, and accuracy, combining systems-level engineering with principled ML.

## How it differs from generic gradient boosting

Three things distinguish the XGBoost formulation from textbook gradient boosting:

1. **Regularization is in the objective, not bolted on.** The per-iteration objective is training loss plus a [[complexity-penalty]] $\gamma T$ on the number of leaves and an [[l2-leaf-penalty]] $\tfrac{1}{2}\lambda \sum w_j^2$ on leaf scores (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).
2. **Second-order Taylor expansion.** Instead of fitting trees to the negative gradient alone, XGBoost expands the loss to second order, using gradients $g_i$ and Hessians $h_i$. This gives closed-form optimal leaf weights and a clean [[structure-score]] for evaluating tree structures. See [[second-order-taylor-approximation]].
3. **Custom losses via $(g_i, h_i)$.** Because the per-iteration objective only depends on $g_i$ and $h_i$, the same solver handles any twice-differentiable loss — regression, logistic, pairwise ranking — by plugging in different gradient and Hessian functions (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## The objective

For an ensemble of $K$ trees $\hat{y}_i = \sum_k f_k(x_i)$, the objective is:

$$\text{obj}(\theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \omega(f_k)$$

with $\omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$.

At step $t$, after Taylor expansion, the new tree minimizes:

$$\sum_i [g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)] + \omega(f_t)$$

For a fixed tree structure, optimal leaf weights are $w_j^* = -G_j / (H_j + \lambda)$, where $G_j$ and $H_j$ are sums of $g_i$ and $h_i$ over instances in leaf $j$. The resulting [[structure-score]] $-\tfrac{1}{2}\sum_j G_j^2 / (H_j + \lambda) + \gamma T$ measures how good a tree structure is (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Tree learning

Tree structures are built greedily by [[split-finding]]: at each candidate split, gain is computed using the structure-score formula. If gain is below $\gamma$, the split is skipped — a principled form of pruning (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Relation to random forests

Tree ensembles are the model used by *both* random forests and gradient boosted trees. The difference is in training: bagging vs. additive boosting (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). See [[decision-tree-ensembles]].

## Born-again decision tree

Because XGBoost ensembles are hard to interpret, model-compression techniques can approximate the full ensemble with a single "born-again" decision tree (source: Gradient boosting.md).

## Related pages

- [[gradient-boosting]]
- [[gradient-tree-boosting]]
- [[second-order-taylor-approximation]]
- [[structure-score]]
- [[split-finding]]
- [[objective-function]]
- [[complexity-penalty]]
- [[l2-leaf-penalty]]
- [[additive-training]]
- [[decision-tree-ensembles]]
