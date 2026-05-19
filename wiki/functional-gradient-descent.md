# Functional Gradient Descent

**Summary**: The view of boosting as iterative gradient descent in *function space* rather than parameter space. Each iteration adds a function in the direction of the negative loss gradient.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## The idea

In ordinary gradient descent, parameters are updated against the gradient of a loss with respect to parameters. In functional gradient descent, the *function itself* is the optimization variable. The update

$$F_m(x) = F_{m-1}(x) - \gamma \sum_{i=1}^{n} \nabla_{F_{m-1}} L(y_i, F_{m-1}(x_i))$$

moves the function $F$ by a small step in the direction that most decreases the empirical loss (source: Gradient boosting.md). The step length $\gamma$ is found via line search.

## Why it matters

Mason, Baxter, Bartlett, and Frean (1999) introduced this view in *Boosting Algorithms as Gradient Descent* (source: Gradient boosting.md). It generalizes boosting beyond regression and classification, since *any* differentiable loss admits a gradient and therefore a descent step.

## Discrete approximation: fit a tree to the gradient

Because the function class $\mathcal{H}$ of trees is discrete, we cannot literally move in the gradient direction. Instead, [[gradient-boosting]] fits the closest available weak learner — i.e., the tree that best predicts the [[pseudo-residuals]]. The line search then finds the multiplier $\gamma_m$ that minimizes the loss along that direction (source: Gradient boosting.md):

$$\gamma_m = \arg\min_\gamma \sum_i L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)).$$

## Proof sketch

A first-order Taylor expansion of $L(y_i, F_{m-1}(x_i) + h_m(x_i))$ around $F_{m-1}(x_i)$, differentiated with respect to $h_m(x_i)$, leaves only the gradient term. The direction of steepest *ascent* is the gradient, so steepest descent goes in the negative direction (source: Gradient boosting.md).

## Related pages

- [[gradient-boosting]]
- [[pseudo-residuals]]
- [[boosting]]
- [[second-order-taylor-approximation]]
