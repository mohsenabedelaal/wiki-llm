# Source: Introduction to Boosted Trees (XGBoost docs)

**Summary**: Summary of the official XGBoost tutorial *Introduction to Boosted Trees*. Derives boosted trees from first principles of supervised learning, with regularization built into the objective and a second-order Taylor expansion of the loss.

**Sources**: `Introduction to Boosted Trees — xgboost 3.2.1 documentation.md`

**Last updated**: 2026-05-15

---

**Original URL**: https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/model.html

## Structure of the tutorial

1. **Elements of supervised learning**: model + parameters; [[objective-function]] = training loss + regularization; [[bias-variance-tradeoff]].
2. **Decision tree ensembles**: introduces [[cart]] (real scores on leaves, not just labels) and [[decision-tree-ensembles]] as the model used by both random forests and gradient boosted trees.
3. **Tree boosting**: derives the training procedure.
   - [[additive-training]]: lock in previous trees, add one new tree at a time.
   - [[second-order-taylor-approximation]] of the loss → per-step objective in terms of $g_i$ (gradient) and $h_i$ (Hessian).
   - Model complexity defined as $\omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum w_j^2$ — see [[complexity-penalty]] and [[l2-leaf-penalty]].
   - [[structure-score]]: closed-form measure of how good a tree structure is.
   - [[split-finding]]: gain formula and the pruning rule "split only if gain $> \gamma$".
4. **Final words**: XGBoost is the library implementing this; it pairs the principled formulation with systems optimizations for scalability and portability.

## Distinctive ideas from this source

- **Random forests and boosted trees are the same model.** Both are tree ensembles. The difference is only how they are *trained* (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md). A prediction service for one works for both.
- **Regularization in the objective is the point.** "Most tree packages treat regularization less carefully, or simply ignore it." Defining $\Omega$ formally is what lets us "obtain models that perform well in the wild."
- **Custom losses via $(g_i, h_i)$**. The per-step objective depends *only* on $g_i$ and $h_i$ after Taylor expansion. Same solver for regression, logistic, ranking.
- **Pruning falls out of the math.** The rule "split when gain $> \gamma$" isn't a heuristic — it's the consequence of including $\gamma T$ in the [[objective-function]] (source: Introduction to Boosted Trees — xgboost 3.2.1 documentation.md).
- **Limitation: one feature at a time.** Greedy [[split-finding]] can fail when the right answer requires reasoning over multiple feature dimensions simultaneously. The tutorial points to *Can Gradient Boosting Learn Simple Arithmetic?* as a known failure example.

## What this source does *not* cover

- The Wikipedia article's [[functional-gradient-descent]] view.
- Many concrete regularization techniques from the broader literature: [[shrinkage]], [[stochastic-gradient-boosting]], minimum observations per leaf. (XGBoost supports these as parameters but the tutorial focuses on the in-objective regularizers.)
- Empirical guidance on [[tree-size]] (covered in Wikipedia).

## Related wiki pages

- [[xgboost]]
- [[additive-training]]
- [[second-order-taylor-approximation]]
- [[structure-score]]
- [[split-finding]]
- [[objective-function]]
- [[gradient-boosting-wikipedia]] (companion source)
