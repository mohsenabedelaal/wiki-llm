# Split Finding

**Summary**: The greedy procedure XGBoost uses to grow each tree: at every leaf, evaluate candidate splits using a gain formula derived from the [[structure-score]], and split when gain exceeds the complexity penalty.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

## Gain formula

For a leaf with instances split into left $L$ and right $R$ children, the gain from splitting is:

$$\text{Gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

This decomposes as: score on the new left leaf + score on the new right leaf − score on the original leaf − [[complexity-penalty]] $\gamma$ for the extra leaf (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Principled pruning

If gain is less than $\gamma$, **don't split** — adding the branch makes the regularized objective worse. This recovers classical pre-pruning behavior as a direct consequence of the regularized objective, not a heuristic (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Efficient enumeration of splits

For a continuous feature, sort instances by the feature value. A single left-to-right scan accumulates $G_L, H_L$ progressively, and the formula above can be evaluated at every candidate split point. This makes split finding linear in the number of instances per feature (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).

## Limitation: one feature at a time

Because splits are added one at a time and considered one feature at a time, additive tree learning can fail on problems where the right split requires reasoning about multiple feature dimensions simultaneously (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). The XGBoost tutorial cites *Can Gradient Boosting Learn Simple Arithmetic?* as a known failure case.

## Related pages

- [[xgboost]]
- [[structure-score]]
- [[complexity-penalty]]
- [[l2-leaf-penalty]]
- [[cart]]
