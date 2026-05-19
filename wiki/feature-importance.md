# Feature Importance

**Summary**: A score quantifying how much each input feature contributes to a model's predictions. In gradient boosted trees, it is usually computed by aggregating the importance assigned by each base learner.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

## How it's computed

For [[gradient-boosting]], feature importance is "usually based on aggregating the importance function of the base learners" (source: Gradient boosting.md). For a tree-based ensemble using entropy-based splits, the ensemble ranks features by entropy reduction averaged across all trees.

In practice, common variants include:

- **Gain**: total reduction in the loss function attributable to splits on a feature.
- **Cover**: total number of samples affected by splits on a feature, weighted by Hessian.
- **Frequency**: how often a feature is selected as a split.

(The two sources don't enumerate these specifically — the Wikipedia article only references the entropy-based aggregation; the variants above are standard XGBoost reporting modes and would need verification.)

## Caveat

Gradient boosting sacrifices interpretability compared to a single decision tree. Following one decision tree's path is trivial; following hundreds is hard. Feature importance gives an aggregate view but does not fully recover individual-prediction explanations. For more interpretability, the "born-again decision tree" approach compresses an XGBoost ensemble into a single decision tree approximating the same decision function (source: Gradient boosting.md).

## Related pages

- [[gradient-boosting]]
- [[xgboost]]
- [[cart]]
