# Boosting

**Summary**: A family of ensemble methods that combine many [[weak-learner]]s sequentially, with each learner trained to correct the errors of those before it.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Idea

Boosting trains weak learners one at a time. Each new learner focuses on the examples the current ensemble gets wrong. The final prediction is a weighted combination of all learners. The result tends to outperform any individual learner, and gradient-boosted trees usually outperform [[random-forest]]-style bagging methods (source: Gradient boosting.md).

## From boosting to gradient boosting

Leo Breiman observed that boosting can be interpreted as an optimization algorithm on a cost function (source: Gradient boosting.md). Friedman (1999, 2001) formalized this for regression, producing the explicit [[gradient-boosting]] algorithm. Mason, Baxter, Bartlett, and Frean simultaneously developed the more abstract view of [[functional-gradient-descent]] — boosting as iterative descent in function space (source: Gradient boosting.md).

The shift from classical boosting (fit to residuals) to gradient boosting (fit to [[pseudo-residuals]]) generalizes the method to any differentiable [[objective-function]].

## Random forests are different

Random forests also combine many trees, but train them in parallel on bootstrap samples (bagging). Boosting is sequential and additive. See [[decision-tree-ensembles]] — random forests and gradient-boosted trees share the same *model* but differ in *training* (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[gradient-boosting]]
- [[weak-learner]]
- [[functional-gradient-descent]]
- [[pseudo-residuals]]
- [[decision-tree-ensembles]]
