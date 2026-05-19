# Shrinkage (Learning Rate)

**Summary**: A multiplier $0 < \nu \leq 1$ applied to each new tree's contribution. Small $\nu$ improves generalization at the cost of needing more iterations.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Update rule

The shrunken update is

$$F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x), \quad 0 < \nu \leq 1$$

(source: Gradient boosting.md). The parameter $\nu$ is the **learning rate** — in XGBoost it's called `eta`.

## Why it works

Empirically, small learning rates (e.g., $\nu < 0.1$) yield dramatic improvements in generalization compared to $\nu = 1$ (no shrinkage). The intuition: each tree fits the [[pseudo-residuals]] of the *current* ensemble, which is a noisy target. Taking small steps reduces overshoot and effectively averages over more "directions" in function space (source: Gradient boosting.md).

## Tradeoff

Smaller $\nu$ requires more iterations $M$ to achieve the same training fit, raising both training time and prediction time (source: Gradient boosting.md). A common practical recipe: fix a small $\nu$, then choose $M$ by early stopping on validation error.

## Related pages

- [[regularization]]
- [[gradient-boosting]]
- [[xgboost]]
