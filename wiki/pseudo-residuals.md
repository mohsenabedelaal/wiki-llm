# Pseudo-residuals

**Summary**: The negative gradient of the loss with respect to the current prediction. Pseudo-residuals are the targets that each new weak learner in [[gradient-boosting]] is trained to fit.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## Definition

At stage $m$, the pseudo-residual for example $i$ is

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

(source: Gradient boosting.md). The next weak learner $h_m$ is fit to the dataset $\{(x_i, r_{im})\}$.

## Why "pseudo"

For ordinary squared-error loss, the pseudo-residual reduces to the standard residual $y_i - F_{m-1}(x_i)$ (up to a constant factor of $2/n$). For other losses — logistic, hinge, absolute error — the gradient is *not* a literal residual but plays the same role: it tells the next learner which direction to push each prediction (source: Gradient boosting.md).

This generalization is what lets [[gradient-boosting]] handle arbitrary differentiable losses while keeping the simple "fit a learner to errors, then add" structure.

## Connection to functional gradient descent

Fitting to pseudo-residuals is the discrete analogue of taking a steepest-descent step in function space. See [[functional-gradient-descent]] for the underlying view (source: Gradient boosting.md).

## XGBoost goes further

[[xgboost]] uses *both* the gradient $g_i$ and the Hessian $h_i$ — a [[second-order-taylor-approximation]] of the loss — rather than only the gradient. This yields closed-form optimal leaf weights and a cleaner regularized objective (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[gradient-boosting]]
- [[functional-gradient-descent]]
- [[second-order-taylor-approximation]]
- [[objective-function]]
