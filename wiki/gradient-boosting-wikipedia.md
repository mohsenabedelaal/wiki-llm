# Source: Gradient Boosting (Wikipedia)

**Summary**: Summary of the Wikipedia article *Gradient boosting*. The article covers the history, generic algorithm, gradient tree boosting, and regularization techniques.

**Sources**: `Gradient boosting.md`

**Last updated**: 2026-05-15

---

**Original URL**: https://en.wikipedia.org/wiki/Gradient_boosting

## What the article covers

- **History**: Breiman's observation that boosting is optimization on a cost function; Friedman's regression-specific formalization (1999, 2001); Mason/Baxter/Bartlett/Frean's [[functional-gradient-descent]] view.
- **Informal introduction**: Builds intuition by deriving the gradient boosting update for squared-error regression, then generalizing — showing that ordinary residuals equal pseudo-residuals (up to a constant) for MSE loss. See [[pseudo-residuals]].
- **General algorithm**: The full pseudo-code: initialize with a constant, then for each stage compute pseudo-residuals, fit a base learner, line-search the multiplier, update. See [[gradient-boosting]].
- **Gradient tree boosting / TreeBoost**: Friedman's modification using fixed-size [[cart]]s with per-region optimal $\gamma_{jm}$. See [[gradient-tree-boosting]].
- **Tree size**: $4 \leq J \leq 8$ is typical and results are insensitive within that range. See [[tree-size]].
- **Regularization techniques**: iteration count $M$, tree depth, [[shrinkage]] (with $\nu < 0.1$ giving "dramatic" improvements), [[stochastic-gradient-boosting]] (with $f \in [0.5, 0.8]$), minimum observations per leaf, complexity penalty on leaf count, $\ell_2$ penalty on leaf values. See [[regularization]].
- **Usage**: Learning-to-rank at Yahoo and Yandex; High Energy Physics (Higgs boson analyses at LHC); geological reservoir-quality evaluation.
- **Names**: GBM (Friedman), MART, BRT, GBM (R `gbm` package), TreeNet.
- **[[feature-importance]]**: aggregated from base learners.
- **Disadvantages**: loss of interpretability; "born-again" decision tree compression can recover some.

## Key claims worth remembering

- Gradient-boosted trees "usually outperform random forest" (source: Gradient boosting.md). This is asserted but not extensively justified in the article.
- The functional-gradient-descent view is what enables boosting to extend to *any* differentiable loss — not just regression and classification but ranking too.
- Many of XGBoost's "innovations" (complexity penalty, L2 on leaves, subsampling) are described in the Wikipedia article as established gradient-boosting regularizers; XGBoost's contribution is to make them central and combine them with a [[second-order-taylor-approximation]] (see the XGBoost tutorial: [[introduction-to-boosted-trees-xgboost]]).

## Related wiki pages

- [[gradient-boosting]] (core concept)
- [[functional-gradient-descent]]
- [[gradient-tree-boosting]]
- [[regularization]] and the regularization-specific pages
- [[introduction-to-boosted-trees-xgboost]] (companion source)
