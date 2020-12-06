# Ranky

#### Compute rankings in Python.

[![Build Status](https://travis-ci.com/Didayolo/ranky.svg?token=sQRwdboThHyw4yYsxjxs&branch=master)](https://travis-ci.com/Didayolo/ranky)

![logo](logo.png)

[Documentation](https://didayolo.github.io/ranky/).


# Get started

```bash
pip install ranky
```
```python
import ranky as rk
```

# Main functions

The main functionalities include **scoring metrics** (e.g. accuracy, roc auc), **rank metrics** (e.g. Kendall Tau, Spearman correlation), **ranking systems** (e.g. Majority judgement, Kemeny-Young method) and some **measurements** (e.g. Kendall's W coefficient of concordance).

Most functions takes as input 2-dimensional `numpy.array` or `pandas.DataFrame` objects. DataFrame are the best to keep track of the names of each data point.

Let's consider the following preference matrix:

![matrix](img/preference_matrix.png)

Each row is a candidate and each column is a judge. Here is the results of `rk.borda(matrix)`, computing the mean rank of each candidate:

![borda](img/borda_example.png) 

We can see that candidate2 has the best average ranking among the four judges.

Let's display it using `rk.show(rk.borda(matrix))`:

![display](img/show_example.png)

## Ranking systems

* Random Dictator. `rk.dictator(m)`
* Score Voting. `rk.score(m)`
* Borda Count. `rk.borda(m)`
* Majority Judgement. `rk.majority(m)`
* Condorcet, p-value Condorcet. `rk.condorcet(m)`, `rk.condorcet(m, wins=rk.p_wins)`

* Center. `rk.center(m)`, `rk.center(m, method='swap')`, etc.
* Kemeny.

## Metrics

* Scoring metrics. `rk.metric(y_true, y_pred, method='accuracy')`

* Rank correlation coefficients. `rk.corr(r1, r2, method='spearman')`

* Rank distances. `rk.dist(r1, r2, method='levenshtein')`


## Visualizations

* `rk.show`, 1D or 2D
* `rk.tsne`, 2D or 3D

## Other

* Consensus
* Concordane
* Centrality
* Kendall's W
