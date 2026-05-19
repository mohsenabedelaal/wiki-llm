# Wiki Log

Append-only record of operations on the wiki.

---

## 2026-05-15 — Initial ingest: gradient boosting + XGBoost intro

**Sources ingested**:
- `raw/Gradient boosting.md` (Wikipedia article on gradient boosting)
- `raw/Introduction to Boosted Trees — xgboost 3.2.1 documentation.md` (Official XGBoost tutorial)

**Pages created (24)**:

Source summaries:
- `gradient-boosting-wikipedia.md`
- `introduction-to-boosted-trees-xgboost.md`

Core concepts:
- `gradient-boosting.md`, `xgboost.md`, `boosting.md`, `weak-learner.md`, `decision-tree-ensembles.md`, `cart.md`

Algorithm internals:
- `pseudo-residuals.md`, `functional-gradient-descent.md`, `additive-training.md`, `gradient-tree-boosting.md`, `second-order-taylor-approximation.md`, `structure-score.md`, `split-finding.md`

Regularization:
- `regularization.md`, `shrinkage.md`, `stochastic-gradient-boosting.md`, `tree-size.md`, `complexity-penalty.md`, `l2-leaf-penalty.md`

Supporting:
- `objective-function.md`, `bias-variance-tradeoff.md`, `feature-importance.md`

Index:
- `index.md`

**Notes**:
- Both sources were ingested in a single integrated pass since the wiki was empty.
- The Wikipedia article provides the general gradient boosting framework; the XGBoost tutorial provides the second-order, in-objective-regularization variant. Pages cross-link wherever the same concept appears in both.
- The `random-forest` link in `boosting.md` is currently dangling (no page yet) — fine to leave as a stub link for future ingest.
- The two terminology systems (Wikipedia uses $J$ for leaves, $\gamma$ for the line-search step; XGBoost uses $T$ for leaves, $\gamma$ for the per-leaf complexity penalty) are noted where they collide — see `cart.md` and `complexity-penalty.md`.
