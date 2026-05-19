# Objective Function

**Summary**: In supervised learning, the function being minimized during training. The XGBoost tutorial frames it as having two parts by construction: training loss + regularization.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Two parts

$$\text{obj}(\theta) = L(\theta) + \Omega(\theta)$$

- **Training loss $L$** — how well the model fits the training data. Examples: mean squared error for regression, logistic loss for classification (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md):

  $$L_{\text{MSE}}(\theta) = \sum_i (y_i - \hat{y}_i)^2$$

  $$L_{\text{logistic}}(\theta) = \sum_i [y_i \ln(1 + e^{-\hat{y}_i}) + (1 - y_i) \ln(1 + e^{\hat{y}_i})]$$

- **Regularization $\Omega$** — controls model complexity to avoid overfitting. "What people usually forget to add." Captures the [[bias-variance-tradeoff]] explicitly (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Why include regularization in the objective

The XGBoost tutorial argues this is the cleaner formulation: optimizing only training loss has no preference between a simple-and-predictive model and an overfit one. Adding $\Omega$ encodes the preference for *simple and predictive* directly in what we minimize (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). It also makes heuristics like pruning fall out as consequences of optimization rather than ad-hoc fixes — see [[split-finding]].

## In XGBoost specifically

For a tree ensemble, the objective is

$$\text{obj}(\theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \omega(f_k)$$

with

$$\omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$$

combining the [[complexity-penalty]] $\gamma T$ and the [[l2-leaf-penalty]] $\tfrac{1}{2}\lambda \sum w_j^2$ (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[regularization]]
- [[bias-variance-tradeoff]]
- [[xgboost]]
- [[additive-training]]
- [[structure-score]]
