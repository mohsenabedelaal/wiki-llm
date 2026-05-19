# CART (Classification and Regression Tree)

**Summary**: A decision tree whose leaves carry real-valued scores, not just class labels. CART is the standard weak learner in [[gradient-tree-boosting]] and the building block of [[xgboost]] ensembles.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`, `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Definition

A CART partitions the input space via axis-aligned splits and assigns a *real score* (not just a class) to each leaf. This makes it more expressive than a classification-only tree: scores can sum across an ensemble, support a principled optimization formulation, and be interpreted as logits, regression outputs, or ranking scores depending on task (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

Formally, a tree $f$ in XGBoost is

$$f(x) = w_{q(x)}, \quad w \in \mathbb{R}^T, \quad q: \mathbb{R}^d \to \{1, \dots, T\}$$

where $q$ maps each input to a leaf index and $w$ is the vector of leaf scores. $T$ is the number of leaves (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

In the Wikipedia formulation, the tree partitions input space into disjoint regions $R_{1m}, \dots, R_{J_m m}$ and predicts a constant $b_{jm}$ in each region (source: Gradient boosting.md):

$$h_m(x) = \sum_{j=1}^{J_m} b_{jm} \mathbf{1}_{R_{jm}}(x).$$

## Why CART (not classification-only trees)

Boosting requires summable, real-valued predictions to compose additively. Classification leaves (which only carry a label) cannot be added across trees. CART scores can. This is what enables the unified optimization view used by [[xgboost]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Tree size

The number of leaves $T$ (or $J$ in Wikipedia notation) controls the maximum order of feature interactions the tree can express. See [[tree-size]].

## Related pages

- [[decision-tree-ensembles]]
- [[gradient-tree-boosting]]
- [[xgboost]]
- [[tree-size]]
- [[weak-learner]]
- [[structure-score]]
