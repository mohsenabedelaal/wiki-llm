# Gradient Boosting

**Summary**: A machine learning technique that builds an ensemble of [[weak-learner]]s in stages, fitting each new learner to the negative gradient of the loss function ([[pseudo-residuals]]). It generalizes traditional boosting to any differentiable loss.

**Sources**: `Gradient boosting.md`, `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Core idea

Gradient boosting builds a prediction model as a sum of weak prediction models — typically [[cart]] trees — added one at a time. Each new model is fit to correct the errors of the current ensemble. The "gradient" part is that the errors being corrected are the *negative gradients of the loss function* with respect to the current predictions, called [[pseudo-residuals]] (source: Gradient boosting.md).

This generalizes traditional residual-fitting boosting. With squared-error loss, the pseudo-residual equals the ordinary residual $y_i - F(x_i)$, but for any other differentiable loss the gradient still gives a valid descent direction. This view is called [[functional-gradient-descent]]: each iteration takes a steepest-descent step in function space (source: Gradient boosting.md).

## Algorithm sketch

Given training data $\{(x_i, y_i)\}$, a differentiable loss $L(y, F(x))$, and $M$ iterations (source: Gradient boosting.md):

1. Initialize $F_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$.
2. For $m = 1$ to $M$:
   1. Compute pseudo-residuals $r_{im} = -\partial L / \partial F(x_i)$ at $F = F_{m-1}$.
   2. Fit a weak learner $h_m(x)$ to $(x_i, r_{im})$.
   3. Find step length $\gamma_m$ by line search.
   4. Update $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$.
3. Return $F_M(x)$.

When the weak learner is a tree, this becomes [[gradient-tree-boosting]] — the form that dominates practical use.

## Connection to XGBoost

[[xgboost]] is a specific implementation of gradient boosted trees with two distinctive choices: a second-order Taylor approximation of the loss ([[second-order-taylor-approximation]]) and a regularization term built directly into the per-iteration objective ([[objective-function]], [[complexity-penalty]], [[l2-leaf-penalty]]) (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Regularization

Without controls, gradient boosting easily overfits. See [[regularization]] for the full toolkit:

- [[shrinkage]] (learning rate $\nu$)
- [[stochastic-gradient-boosting]] (row subsampling)
- [[tree-size]] (number of leaves $J$)
- minimum observations per leaf
- complexity penalties on leaves

## Names

The method appears under many names: Gradient Boosting Machine (GBM, Friedman), Multiple Additive Regression Trees (MART), Boosted Regression Trees (BRT), TreeNet, and "Generalized Boosting Model" in the R `gbm` package (source: Gradient boosting.md).

## Related pages

- [[xgboost]]
- [[boosting]]
- [[pseudo-residuals]]
- [[functional-gradient-descent]]
- [[gradient-tree-boosting]]
- [[regularization]]
- [[weak-learner]]
- [[feature-importance]]
