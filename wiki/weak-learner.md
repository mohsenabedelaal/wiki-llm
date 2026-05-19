# Weak Learner

**Summary**: A simple prediction model that makes few assumptions about the data and only needs to be slightly better than chance. Boosting combines many weak learners into a strong one.

**Sources**: `Gradient boosting.md`, `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

In [[gradient-boosting]], a weak learner (also called a *base learner*) is any function from a class $\mathcal{H}$ used as an additive component of the ensemble. Typical weak learners are shallow decision trees — often [[cart]]s with 4–8 leaves (source: Gradient boosting.md). See [[tree-size]] for guidance.

The defining property is that an individual weak learner has limited capacity; the strength comes from combining many of them via the additive update

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x).$$

When the weak learner is a decision tree, the algorithm is called *gradient-boosted trees* and usually outperforms random forest (source: Gradient boosting.md).

## Decision stumps and interaction depth

A decision stump is a tree with $J=2$ leaves — the simplest non-trivial tree. With stumps, the model can capture main effects but no interactions. Larger $J$ allows higher-order interactions. See [[tree-size]] (source: Gradient boosting.md).

## Related pages

- [[gradient-boosting]]
- [[boosting]]
- [[cart]]
- [[tree-size]]
- [[gradient-tree-boosting]]
