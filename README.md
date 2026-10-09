# Ranky

#### Compute rankings in Python.

[![tests](https://github.com/didayolo/ranky/actions/workflows/tests.yml/badge.svg)](https://github.com/didayolo/ranky/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/ranky)](https://pypi.org/project/ranky/)

![logo](https://raw.githubusercontent.com/didayolo/ranky/master/logo.png)

# Get started

```bash
pip install ranky
```
```python
import ranky as rk
```

Read the **[documentation](https://didayolo.github.io/ranky/)**.

# Main functions

The main functionalities include **scoring metrics** (e.g. accuracy, roc auc), **rank metrics** (e.g. Kendall Tau, Spearman correlation), **ranking systems** (e.g. Majority judgement, Kemeny-Young method) and some **measurements** (e.g. Kendall's W coefficient of concordance).

Most functions take as input 2-dimensional `numpy.array` or `pandas.DataFrame` objects. DataFrames are the best way to keep track of the names of each data point.

Let's consider the following preference matrix:

![matrix](https://raw.githubusercontent.com/didayolo/ranky/master/img/preference_matrix.png)

Each row is a candidate and each column is a judge. Here are the results of `rk.average_rank(matrix)`, computing the mean rank of each candidate:

![borda](https://raw.githubusercontent.com/didayolo/ranky/master/img/borda_example.png)

We can see that candidate2 has the best average rank among the four judges.

Let's display it using `rk.show(rk.average_rank(matrix))`:

![display](https://raw.githubusercontent.com/didayolo/ranky/master/img/show_example.png)

In code:

```python
import pandas as pd
import ranky as rk

m = pd.read_csv('data/matrix.csv', index_col=0)  # candidates x judges
rk.average_rank(m)         # Borda count, lower is better
rk.pairwise(m)             # Copeland's method, higher is better
rk.kendall_w(m, axis=1)    # agreement between judges
rk.show(rk.average_rank(m))
```

## Conventions

* A preference matrix has one row per **candidate** and one column per **judge**. A judge can be a voter, a dataset, a metric, etc.
* By default, **higher scores are better**. Many functions have a `reverse` argument to change this.
* Ranking systems take an `axis` argument, which is the axis of the **judges** (`axis=1` by default). Some lower-level functions (`rk.rank`, `rk.consensus`, `rk.evolution_strategy`, `rk.brute_force`) take the axis of the **candidates** instead. Check the docstrings when in doubt.
* With a `pandas.DataFrame` as input, the results keep the names of the candidates.


## Ranking systems

The rank aggregation methods available include:

* Random Dictator: `rk.dictator(m)`
* Score Voting (mean): `rk.score(m)`
* Borda Count (average rank): `rk.average_rank(m)` or `rk.borda(m)`
* Majority Judgement (median): `rk.majority(m)`
* Uninominal Voting (multi-turn instant-runoff): `rk.uninominal(m, turns=2)`
* Pairwise methods. Copeland's method: `rk.pairwise(m)` or `rk.copeland(m)`, Success rate: `rk.pairwise(m, wins=rk.success_rate)` and more. You can specify your own "wins" function or select one from the `rk.duel` module.
* **Optimal rank aggregation** using any rank metric: `rk.center(m)`, `rk.center(m, method='kendalltau')`. Solver used \[1\].
* _(Kemeny-Young method is optimal rank aggregation using Kendall's tau as metric: `rk.kemeny_young(m)`.)_
* _(Optimal rank aggregation using Spearman correlation as metric is equivalent to Borda count.)_


## Metrics

You can use `rk.any_metric(a, b, method)` to call a metric from **any** of the three categories below.

* **Scoring metrics**: `rk.metric(y_true, y_pred, method='accuracy')`. Methods include: `['accuracy', 'balanced_accuracy', 'precision', 'average_precision', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse', 'mae', 'sar']`

* **Rank correlation coefficients**: `rk.corr(r1, r2, method='spearman')`. Methods include: `['kendalltau', 'spearman', 'pearson']`

* **Rank distances**: `rk.dist(r1, r2, method='levenshtein')`. Methods include: `['hamming', 'levenshtein', 'kendall', 'euclidean', 'winner', 'winner_mistake', 'winner_distance', 'symmetrical_winner_distance']`


_To add: general edit distances, kemeny distance, regression metrics..._


## Visualizations

* Use `rk.show` to visualize preference matrix (2D) or ranking ballots (1D).

`>>> rk.show(m)`

![show example 1](https://raw.githubusercontent.com/didayolo/ranky/master/img/show_example_1.png)

`>>> rk.show(m['judge1'])`

![show example 2](https://raw.githubusercontent.com/didayolo/ranky/master/img/show_example_2.png)

* Use `rk.mds` to visualize (in 2D or 3D) the points in a given metric space. _See `rk.scatterplot` documentation for display arguments._

`>>> rk.mds(m, method='euclidean')`

![MDS example 1](https://raw.githubusercontent.com/didayolo/ranky/master/img/mds_example_1.png)

`>>> rk.mds(m, method='spearman', axis=1)`

![MDS example 2](https://raw.githubusercontent.com/didayolo/ranky/master/img/mds_example_2.png)

* You can use `rk.tsne` similarly to `rk.mds`.

* Use `rk.critical_difference` to plot a [critical difference diagram](https://github.com/mbatchkarov/critical_difference), comparing candidates' performance and grouping them by statistical equivalence. Such diagrams can be seen in \[2, 3\].

`>>> rk.critical_difference(m, comparison_func=rk.bayes_wins)`

![Critical difference example](https://raw.githubusercontent.com/didayolo/ranky/master/img/critical_difference_example.png)

* Show Condorcet graphs using `rk.show_graph(graph)`, based on \[4\]. The graph can be computed with `rk.pairwise(m, return_graph=True)`.


## Other

* Rank, `rk.rank`, convert a 1D score ballot into a ranking.
* Bootstrap, `rk.bootstrap`, sample a given axis.
* Consensus, `rk.consensus`, check if rankings exactly agree.
* Concordance, `rk.concordance`, mean rank correlation between all judges of a preference matrix.
* Centrality, `rk.centrality`, mean rank correlation (or distance) between a ranking and a preference matrix.
* Kendall's W, `rk.kendall_w`, coefficient of concordance.
* Judge generators, `rk.SwapGenerator` and `rk.GaussianGenerator`, sample noisy judges from a reference ranking.
* Utility: `rk.read_codalab_csv` to parse a CSV generated by Codalab representing a leaderboard into a `pandas.DataFrame`.


# Development

See [CONTRIBUTING.md](https://github.com/didayolo/ranky/blob/master/CONTRIBUTING.md) to run the tests, build the documentation and release a new version. The changes between versions are listed in [CHANGELOG.md](https://github.com/didayolo/ranky/blob/master/CHANGELOG.md).


# References

Please cite ranky in your publications if this is useful for your research. Here is an example BibTeX entry:

```
@misc{pavao2020ranky,
  title={ranky},
  author={Adrien Pavao},
  year={2020},
  howpublished={\url{https://github.com/didayolo/ranky}},
}
```

\[1\] Storn R. and Price K., Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.

\[2\] Janez Demsar, Statistical Comparisons of Classifiers over Multiple Data Sets, 7(Jan):1--30, 2006.

\[3\] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, P. Muller, Deep learning for time series classification: a review, Data Mining and Knowledge Discovery, 2018.

\[4\] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008.


# License

Copyright (c) 2020-2021, Adrien PAVAO. This software is released under the Apache License 2.0 (the "License"); you may not use the software except in compliance with the License.

The text of the Apache License 2.0 can be found online at: http://www.opensource.org/licenses/apache2.0.php
