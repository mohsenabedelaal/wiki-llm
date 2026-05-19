---
title: "Gradient boosting"
source: "https://en.wikipedia.org/wiki/Gradient_boosting"
author:
  - "[[Wikipedia]]"
published: 2010-03-22
created: 2026-05-15
description:
tags:
  - "clippings"
---
**Gradient boosting** is a [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning") technique based on [boosting](https://en.wikipedia.org/wiki/Boosting_\(machine_learning\) "Boosting (machine learning)") in a functional space, where the target is *pseudo-residuals* instead of [residuals](https://en.wikipedia.org/wiki/Residuals_\(statistics\) "Residuals (statistics)") as in traditional boosting. It gives a prediction model in the form of an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning "Ensemble learning") of weak prediction models, i.e., models that make very few assumptions about the data, which are typically simple [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning "Decision tree learning").[^1] [^2] When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually outperforms [random forest](https://en.wikipedia.org/wiki/Random_forest "Random forest").[^1] As with other [boosting](https://en.wikipedia.org/wiki/Boosting_\(machine_learning\) "Boosting (machine learning)") methods, a gradient-boosted trees model is built in stages, but it generalizes the other methods by allowing optimization of an arbitrary [differentiable](https://en.wikipedia.org/wiki/Differentiable_function "Differentiable function") [loss function](https://en.wikipedia.org/wiki/Loss_function "Loss function").

## History

The idea of gradient boosting originated in the observation by [Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman "Leo Breiman") that boosting can be interpreted as an optimization algorithm on a suitable cost function.[^3] Explicit regression gradient boosting algorithms were subsequently developed, by [Jerome H. Friedman](https://en.wikipedia.org/wiki/Jerome_H._Friedman "Jerome H. Friedman"),[^4] [^2] (in 1999 and later in 2001) simultaneously with the more general functional gradient boosting perspective of Llew Mason, Jonathan Baxter, Peter Bartlett and Marcus Frean.[^5] [^6] The latter two papers introduced the view of boosting algorithms as iterative *functional gradient descent* algorithms. That is, algorithms that optimize a cost function over function space by iteratively choosing a function (weak hypothesis) that points in the negative gradient direction. This functional gradient view of boosting has led to the development of boosting algorithms in many areas of machine learning and statistics beyond regression and classification.

## Informal introduction

(This section follows the exposition by Cheng Li.[^7])

Like other boosting methods, gradient boosting combines weak "learners" into a single strong learner iteratively. It is easiest to explain in the [least-squares](https://en.wikipedia.org/wiki/Least_squares "Least squares") [regression](https://en.wikipedia.org/wiki/Regression_analysis "Regression analysis") setting, where the goal is to teach a model ${\displaystyle F}$ to predict values of the form ${\displaystyle {\hat {y}}=F(x)}$ by minimizing the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error "Mean squared error") ${\displaystyle {\tfrac {1}{n}}\sum _{i}({\hat {y}}_{i}-y_{i})^{2}}$, where ${\displaystyle i}$ indexes over some training set of size ${\displaystyle n}$ of actual values of the output variable ${\displaystyle y}$:

- ${\displaystyle {\hat {y}}_{i}=}$ the predicted value ${\displaystyle F(x_{i})}$
- ${\displaystyle y_{i}=}$ the observed value
- ${\displaystyle n=}$ the size of the sample, i.e. the number of observations in ${\displaystyle y}$

If the algorithm has ${\displaystyle M}$ stages, at each stage ${\displaystyle m}$ (${\displaystyle 1\leq m\leq M}$), suppose some imperfect model ${\displaystyle F_{m}}$ (for low ${\displaystyle m}$, this model may simply predict ${\displaystyle {\hat {y}}_{i}}$ to be ${\displaystyle {\bar {y}}}$, the mean of ${\displaystyle y}$). In order to improve ${\displaystyle F_{m}}$, our algorithm should add some new estimator, ${\displaystyle h_{m}(x)}$. Thus,

${\displaystyle F_{m+1}(x_{i})=F_{m}(x_{i})+h_{m}(x_{i})=y_{i}}$

or, equivalently,

${\displaystyle h_{m}(x_{i})=y_{i}-F_{m}(x_{i}).}$

Therefore, gradient boosting will fit ${\displaystyle h_{m}}$ to the *[residual](https://en.wikipedia.org/wiki/Errors_and_residuals "Errors and residuals")* ${\displaystyle y_{i}-F_{m}(x_{i})}$. As in other boosting variants, each ${\displaystyle F_{m+1}}$ attempts to correct the errors of its predecessor ${\displaystyle F_{m}}$. A generalization of this idea to [loss functions](https://en.wikipedia.org/wiki/Loss_function "Loss function") other than squared error, and to [classification and ranking problems](https://en.wikipedia.org/wiki/Learning_to_rank "Learning to rank"), follows from the observation that residuals ${\displaystyle h_{m}(x_{i})}$ for a given model are proportional to the negative gradients of the [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error "Mean squared error") loss function (with respect to ${\displaystyle F(x_{i})}$):

${\displaystyle L_{\rm {MSE}}={\frac {1}{n}}\sum _{i=1}^{n}\left(y_{i}-F(x_{i})\right)^{2}}$

${\displaystyle -{\frac {\partial L_{\rm {MSE}}}{\partial F(x_{i})}}={\frac {2}{n}}(y_{i}-F(x_{i}))={\frac {2}{n}}h_{m}(x_{i}).}$

So, gradient boosting could be generalized to a [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent "Gradient descent") algorithm by plugging in a different loss and its gradient.

## Algorithm

Many [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning "Supervised learning") problems involve an output variable y and a vector of input variables x, related to each other with some probabilistic distribution. The goal is to find some function ${\displaystyle {\hat {F}}(x)}$ that best approximates the output variable from the values of input variables. This is formalized by introducing some [loss function](https://en.wikipedia.org/wiki/Loss_function "Loss function") ${\displaystyle L(y,F(x))}$ and minimizing it in expectation:

${\displaystyle {\hat {F}}=\operatorname {argmin} \limits _{F}\mathbb {E} _{x,y}[L(y,F(x))].}$

The gradient boosting method assumes a real-valued y. It seeks an approximation ${\displaystyle {\hat {F}}(x)}$ in the form of a weighted sum of M functions ${\displaystyle h_{m}(x)}$ from some class ${\displaystyle {\mathcal {H}}}$, called base (or [weak](https://en.wikipedia.org/wiki/Weak_learner "Weak learner")) learners:

${\displaystyle {\hat {F}}(x)=\sum _{m=1}^{M}\gamma _{m}h_{m}(x)+{\mbox{const}},}$

where ${\displaystyle \gamma _{m}}$ is the weight at stage ${\displaystyle m}$. We are usually given a training set ${\displaystyle \{(x_{1},y_{1}),\dots ,(x_{n},y_{n})\}}$ of known values of x and corresponding values of y. In accordance with the [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization "Empirical risk minimization") principle, the method tries to find an approximation ${\displaystyle {\hat {F}}(x)}$ that minimizes the average value of the loss function on the training set, i.e., minimizes the empirical risk. It does so by starting with a model, consisting of a constant function ${\displaystyle F_{0}(x)}$, and incrementally expands it in a [greedy](https://en.wikipedia.org/wiki/Greedy_algorithm "Greedy algorithm") fashion:

${\displaystyle F_{0}(x)={\underset {h_{m}\in {\mathcal {H}}}{\arg \min }}\sum _{i=1}^{n}{L(y_{i},h_{m}(x_{i}))},}$

${\displaystyle F_{m}(x)=F_{m-1}(x)+\left({\underset {h_{m}\in {\mathcal {H}}}{\operatorname {arg\,min} }}\left[\sum _{i=1}^{n}L(y_{i},F_{m-1}(x_{i})+h_{m}(x_{i}))\right]\right)(x),}$

for ${\displaystyle m\geq 1}$, where ${\displaystyle h_{m}\in {\mathcal {H}}}$ is a base learner function.

Unfortunately, choosing the best function ${\displaystyle h_{m}}$ at each step for an arbitrary loss function L is a computationally infeasible optimization problem in general. Therefore, we restrict our approach to a simplified version of the problem. The idea is to apply a [steepest descent](https://en.wikipedia.org/wiki/Steepest_descent "Steepest descent") step to this minimization problem (functional gradient descent). The basic idea is to find a local minimum of the loss function by iterating on ${\displaystyle F_{m-1}(x)}$. In fact, the local maximum-descent direction of the loss function is the negative gradient.[^8] Hence, moving a small amount ${\displaystyle \gamma }$ such that the linear approximation remains valid:

${\displaystyle F_{m}(x)=F_{m-1}(x)-\gamma \sum _{i=1}^{n}\nabla _{F_{m-1}}L(y_{i},F_{m-1}(x_{i}))}$

where ${\displaystyle \gamma >0}$. For small ${\displaystyle \gamma }$, this implies that ${\displaystyle L(y_{i},F_{m}(x_{i}))\leq L(y_{i},F_{m-1}(x_{i}))}$.

| Proof of functional form of derivative |
| --- |
| To prove the following, consider the objective  ${\displaystyle O=\sum _{i=1}^{n}L(y_{i},F_{m-1}(x_{i})+h_{m}(x_{i}))}$  Doing a Taylor expansion around the fixed point ${\displaystyle F_{m-1}(x_{i})}$ up to first order  ${\displaystyle O=\sum _{i=1}^{n}L(y_{i},F_{m-1}(x_{i})+h_{m}(x_{i}))\approx \sum _{i=1}^{n}L(y_{i},F_{m-1}(x_{i}))+h_{m}(x_{i})\nabla _{F_{m-1}L(y_{i},F_{m-1}(x_{i}))}+\cdots }$  Now differentiating w.r.t. ${\displaystyle h_{m}(x_{i})}$, only the derivative of the second term remains ${\displaystyle \nabla _{F_{m-1}}L(y_{i},F_{m-1}(x_{i}))}$. This is the direction of steepest ascent and hence we must move in the opposite (i.e., negative) direction in order to move in the direction of steepest descent. |

Furthermore, we can optimize ${\displaystyle \gamma }$ by finding the ${\displaystyle \gamma }$ value for which the loss function has a minimum:

${\displaystyle \gamma _{m}={\underset {\gamma }{\operatorname {argmin} }}\sum _{i=1}^{n}L(y_{i},F_{m}(x_{i}))={\underset {\gamma }{\arg \min }}{\sum _{i=1}^{n}L\left(y_{i},F_{m-1}(x_{i})-\gamma \nabla _{F_{m-1}}L(y_{i},F_{m-1}(x_{i}))\right)}.}$

If we considered the continuous case, i.e., where ${\displaystyle {\mathcal {H}}}$ is the set of arbitrary differentiable functions on ${\displaystyle \mathbb {R} }$, we would update the model in accordance with the following equations

${\displaystyle F_{m}(x)=F_{m-1}(x)-\gamma _{m}\sum _{i=1}^{n}{\nabla _{F_{m-1}}L(y_{i},F_{m-1}(x_{i}))}}$

where ${\displaystyle \gamma _{m}}$ is the step length, defined as 
$$
{\displaystyle \gamma _{m}={\underset {\gamma }{\arg \min }}\sum _{i=1}^{n}L\left(y_{i},F_{m-1}(x_{i})-\gamma \nabla _{F_{m-1}}L(y_{i},F_{m-1}(x_{i}))\right).}
$$
 In the discrete case however, i.e. when the set ${\displaystyle {\mathcal {H}}}$ is finite, we choose the candidate function h closest to the gradient of L for which the coefficient γ may then be calculated with the aid of [line search](https://en.wikipedia.org/wiki/Line_search "Line search") on the above equations. Note that this approach is a heuristic and therefore doesn't yield an exact solution to the given problem, but rather an approximation. In pseudocode, the generic gradient boosting method is:[^4] [^1]

Input: training set ${\displaystyle \{(x_{i},y_{i})\}_{i=1}^{n},}$ a differentiable loss function ${\displaystyle L(y,F(x)),}$ number of iterations M.

Algorithm:

1. Initialize model with a constant value:
	${\displaystyle F_{0}(x)={\underset {\gamma }{\arg \min }}\sum _{i=1}^{n}L(y_{i},\gamma ).}$
2. For m = 1 to M:
	1. Compute so-called *pseudo-residuals*:
		${\displaystyle r_{im}=-\left[{\frac {\partial L(y_{i},F(x_{i}))}{\partial F(x_{i})}}\right]_{F(x)=F_{m-1}(x)}\quad {\text{for }}i=1,\ldots ,n.}$
		2. Fit a base learner (or weak learner, e.g. tree) closed under scaling ${\displaystyle h_{m}(x)}$ to pseudo-residuals, i.e. train it using the training set ${\displaystyle \{(x_{i},r_{im})\}_{i=1}^{n}}$.
		3. Compute multiplier ${\displaystyle \gamma _{m}}$ by solving the following one-dimensional optimization problem:
		${\displaystyle \gamma _{m}={\underset {\gamma }{\operatorname {argmin} }}\sum _{i=1}^{n}L\left(y_{i},F_{m-1}(x_{i})+\gamma h_{m}(x_{i})\right).}$
		4. Update the model:
		${\displaystyle F_{m}(x)=F_{m-1}(x)+\gamma _{m}h_{m}(x).}$
3. Output ${\displaystyle F_{M}(x).}$

## Gradient tree boosting

Gradient boosting is typically used with [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning "Decision tree learning") (especially [CARTs](https://en.wikipedia.org/wiki/Classification_and_regression_tree "Classification and regression tree")) of a fixed size as base learners. For this special case, Friedman proposes a modification to gradient boosting method which improves the quality of fit of each base learner.

Generic gradient boosting at the *m* -th step would fit a decision tree ${\displaystyle h_{m}(x)}$ to pseudo-residuals. Let ${\displaystyle J_{m}}$ be the number of its leaves. The tree partitions the input space into ${\displaystyle J_{m}}$ disjoint regions ${\displaystyle R_{1m},\ldots ,R_{J_{m}m}}$ and predicts a constant value in each region. Using the [indicator notation](https://en.wikipedia.org/wiki/Indicator_notation "Indicator notation"), the output of ${\displaystyle h_{m}(x)}$ for input *x* can be written as the sum:

${\displaystyle h_{m}(x)=\sum _{j=1}^{J_{m}}b_{jm}\mathbf {1} _{R_{jm}}(x),}$

where ${\displaystyle b_{jm}}$ is the value predicted in the region ${\displaystyle R_{jm}}$.[^9]

Then the coefficients ${\displaystyle b_{jm}}$ are multiplied by some value ${\displaystyle \gamma _{m}}$, chosen using line search so as to minimize the loss function, and the model is updated as follows:

${\displaystyle F_{m}(x)=F_{m-1}(x)+\gamma _{m}h_{m}(x),\quad \gamma _{m}={\underset {\gamma }{\operatorname {arg\,min} }}\sum _{i=1}^{n}L(y_{i},F_{m-1}(x_{i})+\gamma h_{m}(x_{i})).}$

Friedman proposes to modify this algorithm so that it chooses a separate optimal value ${\displaystyle \gamma _{jm}}$ for each of the tree's regions, instead of a single ${\displaystyle \gamma _{m}}$ for the whole tree. He calls the modified algorithm "TreeBoost". The coefficients ${\displaystyle b_{jm}}$ from the tree-fitting procedure can be then simply discarded and the model update rule becomes:

${\displaystyle F_{m}(x)=F_{m-1}(x)+\sum _{j=1}^{J_{m}}\gamma _{jm}\mathbf {1} _{R_{jm}}(x),\quad \gamma _{jm}={\underset {\gamma }{\operatorname {arg\,min} }}\sum _{x_{i}\in R_{jm}}L(y_{i},F_{m-1}(x_{i})+\gamma ).}$

When the loss ${\displaystyle L(\cdot ,\cdot )}$ is mean-squared error (MSE) the coefficients ${\displaystyle \gamma _{jm}}$ coincide with the coefficients of the tree-fitting procedure ${\displaystyle b_{jm}}$.

### Tree size

The number ${\displaystyle J}$ of terminal nodes in the trees is a parameter which controls the maximum allowed level of [interaction](https://en.wikipedia.org/wiki/Interaction_\(statistics\) "Interaction (statistics)") between variables in the model. With ${\displaystyle J=2}$ ([decision stumps](https://en.wikipedia.org/wiki/Decision_stump "Decision stump")), no interaction between variables is allowed. With ${\displaystyle J=3}$ the model may include effects of the interaction between up to two variables, and so on. ${\displaystyle J}$ can be adjusted for a data set at hand.

Hastie et al.[^1] comment that typically ${\displaystyle 4\leq J\leq 8}$ work well for boosting and results are fairly insensitive to the choice of ${\displaystyle J}$ in this range, ${\displaystyle J=2}$ is insufficient for many applications, and ${\displaystyle J>10}$ is unlikely to be required.

## Regularization

Fitting the training set too closely can lead to degradation of the model's generalization ability, that is, its performance on unseen examples. Several so-called [regularization](https://en.wikipedia.org/wiki/Regularization_\(mathematics\) "Regularization (mathematics)") techniques reduce this [overfitting](https://en.wikipedia.org/wiki/Overfitting "Overfitting") effect by constraining the fitting procedure.

One natural regularization parameter is the number of gradient boosting iterations *M* (i.e. the number of base models). Increasing *M* reduces the error on training set, but increases risk of overfitting. An optimal value of *M* is often selected by monitoring prediction error on a separate validation data set.

Another regularization parameter for tree boosting is tree depth. The higher this value the more likely the model will overfit the training data.

### Shrinkage

An important part of gradient boosting is regularization by shrinkage which uses a modified update rule:

${\displaystyle F_{m}(x)=F_{m-1}(x)+\nu \cdot \gamma _{m}h_{m}(x),\quad 0<\nu \leq 1,}$

where [parameter](https://en.wikipedia.org/wiki/Hyperparameter_\(machine_learning\) "Hyperparameter (machine learning)") ${\displaystyle \nu }$ is called the "learning rate".

Empirically, it has been found that using small [learning rates](https://en.wikipedia.org/wiki/Learning_rate "Learning rate") (such as ${\displaystyle \nu <0.1}$) yields dramatic improvements in models' generalization ability over gradient boosting without shrinking (${\displaystyle \nu =1}$).[^1] However, it comes at the price of increasing [computational time](https://en.wikipedia.org/wiki/Computational_time "Computational time") both during training and [querying](https://en.wikipedia.org/wiki/Information_retrieval "Information retrieval"): lower learning rate requires more iterations.

### Stochastic gradient boosting

Soon after the introduction of gradient boosting, Friedman proposed a minor modification to the algorithm, motivated by [Breiman](https://en.wikipedia.org/wiki/Leo_Breiman "Leo Breiman") 's [bootstrap aggregation](https://en.wikipedia.org/wiki/Bootstrap_aggregation "Bootstrap aggregation") ("bagging") method.[^2] Specifically, he proposed that at each iteration of the algorithm, a base learner should be fit on a subsample of the training set drawn at random without replacement.[^10] Friedman observed a substantial improvement in gradient boosting's accuracy with this modification.

Subsample size is some constant fraction ${\displaystyle f}$ of the size of the training set. When ${\displaystyle f=1}$, the algorithm is deterministic and identical to the one described above. Smaller values of ${\displaystyle f}$ introduce randomness into the algorithm and help prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting "Overfitting"), acting as a kind of [regularization](https://en.wikipedia.org/wiki/Regularization_\(mathematics\) "Regularization (mathematics)"). The algorithm also becomes faster, because regression trees have to be fit to smaller datasets at each iteration. Friedman [^2] obtained that ${\displaystyle 0.5\leq f\leq 0.8}$ leads to good results for small and moderate sized training sets. Therefore, ${\displaystyle f}$ is typically set to 0.5, meaning that one half of the training set is used to build each base learner.[^11]

Also, like in bagging, subsampling allows one to define an [out-of-bag error](https://en.wikipedia.org/wiki/Out-of-bag_error "Out-of-bag error") of the prediction performance improvement by evaluating predictions on those observations which were not used in the building of the next base learner. Out-of-bag estimates help avoid the need for an independent validation dataset, but often underestimate actual performance improvement and the optimal number of iterations.[^12] [^13]

### Number of observations in leaves

Gradient tree boosting implementations often also use regularization by limiting the minimum number of observations in trees' terminal nodes. It is used in the tree building process by ignoring any splits that lead to nodes containing fewer than this number of training set instances.

Imposing this limit helps to reduce variance in predictions at leaves.

### Complexity penalty

Another useful regularization technique for gradient boosted model is to penalize its complexity.[^14] For gradient boosted trees, model complexity can be defined as the proportional number of leaves in the trees. The joint optimization of loss and model complexity corresponds to a post-pruning algorithm to remove branches that fail to reduce the loss by a threshold.

Other kinds of regularization such as an ${\displaystyle \ell _{2}}$ penalty on the leaf values can also be used to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting "Overfitting").[^15]

## Usage

Gradient boosting can be used in the field of [learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank "Learning to rank"). The commercial web search engines [Yahoo](https://en.wikipedia.org/wiki/Yahoo "Yahoo") [^16] and [Yandex](https://en.wikipedia.org/wiki/Yandex "Yandex") [^17] use variants of gradient boosting in their machine-learned ranking engines. Gradient boosting is also utilized in High Energy Physics in data analysis. At the Large Hadron Collider (LHC), variants of gradient boosting Deep Neural Networks (DNN) were successful in reproducing the results of non-machine learning methods of analysis on datasets used to discover the [Higgs boson](https://en.wikipedia.org/wiki/Higgs_boson "Higgs boson").[^18] Gradient boosting decision tree was also applied in earth and geological studies – for example quality evaluation of sandstone reservoir.[^19]

## Names

The method goes by a variety of names. Friedman introduced his regression technique as a "Gradient Boosting Machine" (GBM).[^4] Mason, Baxter et al. described the generalized abstract class of algorithms as "functional gradient boosting".[^5] [^6] Friedman et al. describe an advancement of gradient boosted models as Multiple Additive Regression Trees (MART);[^20] Elith et al. describe that approach as "Boosted Regression Trees" (BRT).[^21]

A popular open-source implementation for [R](https://en.wikipedia.org/wiki/R_\(programming_language\) "R (programming language)") calls it a "Generalized Boosting Model",[^12] however packages expanding this work use BRT.[^22] Yet another name is TreeNet, after an early commercial implementation from Salford System's Dan Steinberg, one of researchers who pioneered the use of tree-based methods.[^23]

## Feature importance ranking

Gradient boosting can be used for feature importance ranking, which is usually based on aggregating importance function of the base learners.[^24] For example, if a gradient boosted trees algorithm is developed using entropy-based [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning "Decision tree learning"), the ensemble algorithm ranks the importance of features based on entropy as well with the caveat that it is averaged out over all base learners.[^24] [^1]

## Disadvantages

While boosting can increase the accuracy of a base learner, such as a decision tree or linear regression, it sacrifices intelligibility and [interpretability](https://en.wikipedia.org/wiki/Interpretability "Interpretability").[^24] [^25] For example, following the path that a decision tree takes to make its decision is trivial and self-explained, but following the paths of hundreds or thousands of trees is much harder. To achieve both performance and interpretability, some model compression techniques allow transforming an XGBoost into a single "born-again" decision tree that approximates the same decision function.[^26] Furthermore, its implementation may be more difficult due to the higher computational demand.

[^1]: Hastie, T.; Tibshirani, R.; Friedman, J. H. (2009). ["10. Boosting and Additive Trees"](https://web.archive.org/web/20091110212529/http://www-stat.stanford.edu/~tibs/ElemStatLearn/). *The Elements of Statistical Learning* (2nd ed.). New York: Springer. pp. 337–384. [ISBN](https://en.wikipedia.org/wiki/ISBN_\(identifier\) "ISBN (identifier)") [978-0-387-84857-0](https://en.wikipedia.org/wiki/Special:BookSources/978-0-387-84857-0 "Special:BookSources/978-0-387-84857-0"). Archived from [the original](http://www-stat.stanford.edu/~tibs/ElemStatLearn/) on 2009-11-10.

[^2]: Friedman, J. H. (March 1999). ["Stochastic Gradient Boosting"](https://web.archive.org/web/20140801033113/http://statweb.stanford.edu/~jhf/ftp/stobst.pdf) (PDF). Archived from [the original](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) (PDF) on 2014-08-01. Retrieved 2013-11-13.

[^3]: Breiman, L. (June 1997). ["Arcing The Edge"](https://statistics.berkeley.edu/sites/default/files/tech-reports/486.pdf) (PDF). *Technical Report 486*. Statistics Department, University of California, Berkeley.

[^4]: Friedman, J. H. (February 1999). ["Greedy Function Approximation: A Gradient Boosting Machine"](https://web.archive.org/web/20191101082737/http://statweb.stanford.edu/~jhf/ftp/trebst.pdf) (PDF). Archived from [the original](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) (PDF) on 2019-11-01. Retrieved 2018-08-27.

[^5]: Mason, L.; Baxter, J.; Bartlett, P. L.; Frean, Marcus (1999). ["Boosting Algorithms as Gradient Descent"](http://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf) (PDF). In [S.A. Solla](https://en.wikipedia.org/wiki/Sara_Solla "Sara Solla") and T.K. Leen and K. Müller (ed.). *Advances in Neural Information Processing Systems 12*. MIT Press. pp. 512–518.

[^6]: Mason, L.; Baxter, J.; Bartlett, P. L.; Frean, Marcus (May 1999). ["Boosting Algorithms as Gradient Descent in Function Space"](https://web.archive.org/web/20181222170928/https://www.maths.dur.ac.uk/~dma6kp/pdf/face_recognition/Boosting/Mason99AnyboostLong.pdf) (PDF). Archived from [the original](https://www.maths.dur.ac.uk/~dma6kp/pdf/face_recognition/Boosting/Mason99AnyboostLong.pdf) (PDF) on 2018-12-22.

[^7]: Cheng Li. ["A Gentle Introduction to Gradient Boosting"](http://www.chengli.io/tutorials/gradient_boosting.pdf) (PDF).

[^8]: Lambers, Jim (2011–2012). ["The Method of Steepest Descent"](https://www.math.usm.edu/lambers/mat419/lecture10.pdf) (PDF).

[^9]: Note: in case of usual CART trees, the trees are fitted using least-squares loss, and so the coefficient ${\displaystyle b_{jm}}$ for the region ${\displaystyle R_{jm}}$ is equal to just the value of output variable, averaged over all training instances in ${\displaystyle R_{jm}}$.

[^10]: Note that this is different from bagging, which samples with replacement because it uses samples of the same size as the training set.

[^11]: Arrabi, Nooshin; Torabi, Mohhamadreza; Fassihi, Afshin; Ghasemi, Fahimeh. "Identification of potential vascular endothelial growth factor receptor inhibitors via tree-based learning modeling and molecular docking simulation". *Chemometrics*. **1** (1): 1. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1002/cem.3545](https://doi.org/10.1002%2Fcem.3545).

[^12]: Ridgeway, Greg (2007). [Generalized Boosted Models: A guide to the gbm package.](https://cran.r-project.org/web/packages/gbm/gbm.pdf)

[^13]: [Learn Gradient Boosting Algorithm for better predictions (with codes in R)](https://www.analyticsvidhya.com/blog/2015/09/complete-guide-boosting-methods/)

[^14]: Tianqi Chen. [Introduction to Boosted Trees](https://web.njit.edu/~usman/courses/cs675_fall16/BoostedTree.pdf)

[^15]: Arrabi, Nooshin; Torabi, Mohhamadreza; Fassihi, Afshin; Ghasemi, Fahimeh. "Identification of potential vascular endothelial growth factor receptor inhibitors via tree-based learning modeling and molecular docking simulation". *Chemometrics*. **1** (1): 1. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1002/cem.3545](https://doi.org/10.1002%2Fcem.3545).

[^16]: Cossock, David and Zhang, Tong (2008). [Statistical Analysis of Bayes Optimal Subset Ranking](http://www.stat.rutgers.edu/~tzhang/papers/it08-ranking.pdf) [Archived](https://web.archive.org/web/20100807162855/http://www.stat.rutgers.edu/~tzhang/papers/it08-ranking.pdf) 2010-08-07 at the [Wayback Machine](https://en.wikipedia.org/wiki/Wayback_Machine "Wayback Machine"), page 14.

[^17]: [Yandex corporate blog entry about new ranking model "Snezhinsk"](http://webmaster.ya.ru/replies.xml?item_no=5707&ncrnd=5118) [Archived](https://web.archive.org/web/20120301165959/http://webmaster.ya.ru/replies.xml?item_no=5707&ncrnd=5118) 2012-03-01 at the [Wayback Machine](https://en.wikipedia.org/wiki/Wayback_Machine "Wayback Machine") (in Russian)

[^18]: Lalchand, Vidhi (2020). "Extracting more from boosted decision trees: A high energy physics case study". [arXiv](https://en.wikipedia.org/wiki/ArXiv_\(identifier\) "ArXiv (identifier)"):[2001.06033](https://arxiv.org/abs/2001.06033) \[[stat.ML](https://arxiv.org/archive/stat.ML)\].

[^19]: Ma, Longfei; Xiao, Hanmin; Tao, Jingwei; Zheng, Taiyi; Zhang, Haiqin (1 January 2022). ["An intelligent approach for reservoir quality evaluation in tight sandstone reservoir using gradient boosting decision tree algorithm"](https://doi.org/10.1515%2Fgeo-2022-0354). *Open Geosciences*. **14** (1): 629–645. [Bibcode](https://en.wikipedia.org/wiki/Bibcode_\(identifier\) "Bibcode (identifier)"):[2022OGeo...14..354M](https://ui.adsabs.harvard.edu/abs/2022OGeo...14..354M). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1515/geo-2022-0354](https://doi.org/10.1515%2Fgeo-2022-0354). [ISSN](https://en.wikipedia.org/wiki/ISSN_\(identifier\) "ISSN (identifier)") [2391-5447](https://search.worldcat.org/issn/2391-5447).

[^20]: Friedman, Jerome (2003). "Multiple Additive Regression Trees with Application in Epidemiology". *Statistics in Medicine*. **22** (9): 1365–1381. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1002/sim.1501](https://doi.org/10.1002%2Fsim.1501). [PMID](https://en.wikipedia.org/wiki/PMID_\(identifier\) "PMID (identifier)") [12704603](https://pubmed.ncbi.nlm.nih.gov/12704603). [S2CID](https://en.wikipedia.org/wiki/S2CID_\(identifier\) "S2CID (identifier)") [41965832](https://api.semanticscholar.org/CorpusID:41965832).

[^21]: Elith, Jane (2008). ["A working guide to boosted regression trees"](https://doi.org/10.1111%2Fj.1365-2656.2008.01390.x). *Journal of Animal Ecology*. **77** (4): 802–813. [Bibcode](https://en.wikipedia.org/wiki/Bibcode_\(identifier\) "Bibcode (identifier)"):[2008JAnEc..77..802E](https://ui.adsabs.harvard.edu/abs/2008JAnEc..77..802E). [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1111/j.1365-2656.2008.01390.x](https://doi.org/10.1111%2Fj.1365-2656.2008.01390.x). [PMID](https://en.wikipedia.org/wiki/PMID_\(identifier\) "PMID (identifier)") [18397250](https://pubmed.ncbi.nlm.nih.gov/18397250).

[^22]: Elith, Jane. ["Boosted Regression Trees for ecological modeling"](https://web.archive.org/web/20200725061155/https://cran.r-project.org/web/packages/dismo/vignettes/brt.pdf) (PDF). *CRAN*. Archived from [the original](https://cran.r-project.org/web/packages/dismo/vignettes/brt.pdf) (PDF) on 25 July 2020. Retrieved 31 August 2018.

[^23]: ["Exclusive: Interview with Dan Steinberg, President of Salford Systems, Data Mining Pioneer"](https://www.kdnuggets.com/2013/06/exclusive-interview-dan-steinberg-salford-systems-data-mining-solutions-provider.html). *KDnuggets*.

[^24]: Piryonesi, S. Madeh; El-Diraby, Tamer E. (2020-03-01). ["Data Analytics in Asset Management: Cost-Effective Prediction of the Pavement Condition Index"](https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29IS.1943-555X.0000512). *Journal of Infrastructure Systems*. **26** (1): 04019036. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1061/(ASCE)IS.1943-555X.0000512](https://doi.org/10.1061%2F%28ASCE%29IS.1943-555X.0000512). [ISSN](https://en.wikipedia.org/wiki/ISSN_\(identifier\) "ISSN (identifier)") [1943-555X](https://search.worldcat.org/issn/1943-555X). [S2CID](https://en.wikipedia.org/wiki/S2CID_\(identifier\) "S2CID (identifier)") [213782055](https://api.semanticscholar.org/CorpusID:213782055).

[^25]: Wu, Xindong; Kumar, Vipin; Ross Quinlan, J.; Ghosh, Joydeep; Yang, Qiang; Motoda, Hiroshi; McLachlan, Geoffrey J.; Ng, Angus; Liu, Bing; Yu, Philip S.; Zhou, Zhi-Hua (2008-01-01). "Top 10 algorithms in data mining". *Knowledge and Information Systems*. **14** (1): 1–37. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1007/s10115-007-0114-2](https://doi.org/10.1007%2Fs10115-007-0114-2). [hdl](https://en.wikipedia.org/wiki/Hdl_\(identifier\) "Hdl (identifier)"):[10983/15329](https://hdl.handle.net/10983%2F15329). [ISSN](https://en.wikipedia.org/wiki/ISSN_\(identifier\) "ISSN (identifier)") [0219-3116](https://search.worldcat.org/issn/0219-3116). [S2CID](https://en.wikipedia.org/wiki/S2CID_\(identifier\) "S2CID (identifier)") [2367747](https://api.semanticscholar.org/CorpusID:2367747).

[^26]: Sagi, Omer; Rokach, Lior (2021). "Approximating XGBoost with an interpretable decision tree". *Information Sciences*. **572** (2021): 522–542. [doi](https://en.wikipedia.org/wiki/Doi_\(identifier\) "Doi (identifier)"):[10.1016/j.ins.2021.05.055](https://doi.org/10.1016%2Fj.ins.2021.05.055).