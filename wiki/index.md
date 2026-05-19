# Wiki Index

Table of contents for the XGBoost / gradient boosting wiki.

## Sources

- [gradient-boosting-wikipedia](gradient-boosting-wikipedia.md) — Wikipedia article on gradient boosting (history, algorithm, regularization).
- [introduction-to-boosted-trees-xgboost](introduction-to-boosted-trees-xgboost.md) — Official XGBoost tutorial deriving boosted trees from supervised-learning principles.

## Core concepts

- [gradient-boosting](gradient-boosting.md) — the core technique: ensemble of weak learners fit to pseudo-residuals.
- [xgboost](xgboost.md) — scalable, regularized implementation using a second-order Taylor approximation.
- [boosting](boosting.md) — broader family of sequential ensembles; framing for gradient boosting.
- [weak-learner](weak-learner.md) — base learners (typically shallow trees) combined additively.
- [decision-tree-ensembles](decision-tree-ensembles.md) — the shared model behind random forests and boosted trees.
- [cart](cart.md) — Classification and Regression Trees with real-valued leaf scores.

## Algorithm internals

- [pseudo-residuals](pseudo-residuals.md) — negative gradient of the loss; the per-step fitting target.
- [functional-gradient-descent](functional-gradient-descent.md) — boosting as gradient descent in function space.
- [additive-training](additive-training.md) — fix previous trees, add one new tree at a time.
- [gradient-tree-boosting](gradient-tree-boosting.md) — Friedman's TreeBoost with per-region step size.
- [second-order-taylor-approximation](second-order-taylor-approximation.md) — XGBoost's expansion using gradients and Hessians.
- [structure-score](structure-score.md) — XGBoost's closed-form quality measure for a tree structure.
- [split-finding](split-finding.md) — gain-based greedy split selection and built-in pruning.

## Regularization

- [regularization](regularization.md) — overview of techniques to prevent overfitting in boosting.
- [shrinkage](shrinkage.md) — learning rate $\nu$ scaling each tree's contribution.
- [stochastic-gradient-boosting](stochastic-gradient-boosting.md) — row subsampling each iteration.
- [tree-size](tree-size.md) — number of leaves controls interaction depth.
- [complexity-penalty](complexity-penalty.md) — $\gamma T$ term; produces principled pruning.
- [l2-leaf-penalty](l2-leaf-penalty.md) — $\tfrac{1}{2}\lambda \sum w_j^2$ on leaf scores.

## Supporting concepts

- [objective-function](objective-function.md) — training loss + regularization; the thing being minimized.
- [bias-variance-tradeoff](bias-variance-tradeoff.md) — simple vs. predictive; encoded in the objective.
- [feature-importance](feature-importance.md) — aggregating per-tree importance across the ensemble.

## Operations log

See [log.md](log.md) for an append-only record of wiki changes.
