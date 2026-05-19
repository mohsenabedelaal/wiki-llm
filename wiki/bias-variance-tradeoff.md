# Bias-Variance Tradeoff

**Summary**: The tradeoff between a model being *simple* (high bias, low variance) and being *predictive on the training data* (low bias, high variance). The XGBoost tutorial frames the [[objective-function]] = loss + regularization split as a direct encoding of this tradeoff.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## The idea

When fitting data, we want both:
- **Predictiveness** — low training error.
- **Simplicity** — generalizes to unseen data.

A too-simple model underfits (high bias). A too-complex model overfits (high variance). The right balance lies in between (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

The XGBoost tutorial illustrates this with three candidate step-function fits: only the middle-complexity one is a "reasonable" fit; the others either ignore the data or interpolate noise (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## How it lands in gradient boosting

In [[gradient-boosting]] and [[xgboost]], the tradeoff is managed by [[regularization]]:

- More iterations $M$ → lower bias, higher variance.
- Larger trees / higher [[tree-size]] → lower bias, higher variance.
- [[shrinkage]] with small $\nu$ → effectively reduces per-tree variance.
- [[stochastic-gradient-boosting]] → reduces variance via subsampling.
- [[complexity-penalty]] and [[l2-leaf-penalty]] → directly push toward simpler trees in the [[objective-function]].

## Related pages

- [[regularization]]
- [[objective-function]]
- [[gradient-boosting]]
- [[xgboost]]
