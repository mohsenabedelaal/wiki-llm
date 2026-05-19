# Second-order Taylor Approximation

**Summary**: XGBoost expands the loss to second order around the current prediction, summarizing each example by a gradient $g_i$ and Hessian $h_i$. This yields closed-form optimal leaf weights and a clean tree-evaluation score.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## The expansion

[[additive-training]] adds $f_t(x_i)$ to the previous prediction $\hat{y}_i^{(t-1)}$. A second-order Taylor expansion of $l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))$ around $\hat{y}_i^{(t-1)}$ gives:

$$\text{obj}^{(t)} \approx \sum_i [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)] + \omega(f_t) + \text{constant}$$

where (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md):

$$g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)}), \quad h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)}).$$

Dropping constants, the per-step objective becomes

$$\sum_i [g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)] + \omega(f_t).$$

## Why this matters

1. **One solver for every loss.** The per-step objective depends only on the pair $(g_i, h_i)$. Plug in different loss functions — squared error, logistic, pairwise ranking — and the same XGBoost solver handles them all. This is how XGBoost supports custom losses (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

2. **Closed-form leaf weights.** With the [[l2-leaf-penalty]] $\tfrac{1}{2}\lambda \sum w_j^2$ included, the objective becomes quadratic in each leaf weight $w_j$. Setting the derivative to zero gives the optimal weight directly:

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad G_j = \sum_{i \in I_j} g_i, \quad H_j = \sum_{i \in I_j} h_i.$$

3. **Closed-form tree score.** Plugging $w_j^*$ back yields the [[structure-score]], which measures how good a tree structure is.

## Contrast with generic gradient boosting

Generic [[gradient-boosting]] uses only the first-order term — it fits a tree to the negative gradient ([[pseudo-residuals]]) and finds the step size by line search. XGBoost uses the second-order term to do the line search analytically, leaf by leaf (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## MSE special case

For mean-squared-error loss, the Taylor form is exact: it has a first-order "residual" term and a clean quadratic term, with no approximation error. Other losses (logistic, etc.) don't simplify so nicely, which is why the Taylor trick is used (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[xgboost]]
- [[additive-training]]
- [[structure-score]]
- [[l2-leaf-penalty]]
- [[pseudo-residuals]]
- [[objective-function]]
