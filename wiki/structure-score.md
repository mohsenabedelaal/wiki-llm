# Structure Score

**Summary**: XGBoost's closed-form measure of how good a tree structure is, given the current gradient and Hessian statistics. Analogous to impurity in classical tree learning, but also accounts for [[regularization]].

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Derivation

After the [[second-order-taylor-approximation]], the per-step objective grouped by leaf $j$ is:

$$\text{obj}^{(t)} = \sum_{j=1}^{T} \left[G_j w_j + \tfrac{1}{2}(H_j + \lambda) w_j^2 \right] + \gamma T$$

where $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$ sum gradient and Hessian statistics over the instances $I_j$ assigned to leaf $j$ (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

Each $w_j$ is independent in this expression, so the quadratic in $w_j$ is minimized at

$$w_j^* = -\frac{G_j}{H_j + \lambda}.$$

Substituting back gives the **structure score**:

$$\text{obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T.$$

The lower this value, the better the structure (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Interpretation

"For a given tree structure, push the statistics $g_i, h_i$ to the leaves they belong to, sum them, and use the formula." It's structurally analogous to impurity scoring in standard tree learning — except it also bakes in the [[l2-leaf-penalty]] (via $\lambda$) and the [[complexity-penalty]] (via $\gamma T$) (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Using it: split finding

Because the structure score decomposes additively over leaves, you can score a candidate split by comparing the structure scores of the leaf before and after splitting. This produces the **gain** used in [[split-finding]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Related pages

- [[xgboost]]
- [[second-order-taylor-approximation]]
- [[split-finding]]
- [[l2-leaf-penalty]]
- [[complexity-penalty]]
- [[objective-function]]
