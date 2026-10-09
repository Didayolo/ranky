""" Smoke tests: check that the plots are drawn without error. """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ranky as rk


class TestShow:
    def test_1d(self, M, df):
        rk.show(M[:, 0])
        rk.show(list(M[:, 0]), annot=True, title='t', xlabel='x', ylabel='y')
        rk.show(df['j1'])
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        assert labels == list(df.index)

    def test_2d(self, M, df):
        rk.show(M, annot=True)
        rk.show(df)
        ax = plt.gca()
        np.testing.assert_array_equal(ax.get_xticks(), [0.5, 1.5, 2.5]) # centered on the cells
        assert [t.get_text() for t in ax.get_xticklabels()] == list(df.columns)

    def test_3d(self):
        with pytest.raises(ValueError):
            rk.show(np.zeros((2, 2, 2)))


def test_show_learning_curve():
    rk.show_learning_curve([0.1, 0.5, 0.6])


def test_show_graph(M):
    _, graph = rk.pairwise(M, return_graph=True)
    rk.show_graph(graph)
    rk.show_graph(graph, names=list('abcde'))


class TestScatterplot:
    def test_2d(self):
        rk.scatterplot(np.random.rand(4, 2), names=list('abcd'), legend=True)

    def test_3d(self):
        rk.scatterplot(np.random.rand(4, 3), dim=3, names=list('abcd'))

    def test_bad_dim(self):
        with pytest.raises(ValueError):
            rk.scatterplot(np.random.rand(4, 2), dim=4)


@pytest.fixture
def judges():
    """ 6 candidates x 4 judges, with no constant row or column. """
    np.random.seed(0)
    m = rk.SwapGenerator().fit(6).sample(n=4, N=3) + np.random.rand(6, 4)
    return pd.DataFrame(m, columns=['j1', 'j2', 'j3', 'j4'])


def test_tsne(judges):
    rk.tsne(judges, axis=0)
    rk.tsne(judges, axis=1, dim=3)
    with pytest.raises(ValueError):
        rk.tsne(judges, axis=2)


def test_mds(judges):
    rk.mds(judges, axis=0)
    rk.mds(judges, axis=1, method='euclidean', dim=3)
    rk.mds_from_dist_matrix(rk.distance_matrix(judges, method='euclidean'))
    with pytest.raises(ValueError):
        rk.mds(judges, axis=2)


class TestCriticalDifference:
    def test_default(self, judges):
        rk.critical_difference(judges)
        rk.critical_difference(judges.T, axis=0, comparison_func=rk.bayes_wins)

    def test_kwargs_are_passed(self, judges):
        calls = []

        def comparison(a, b, flag=None):
            calls.append(flag)
            return False

        rk.critical_difference(judges, comparison_func=comparison, flag='x')
        assert calls and all(flag == 'x' for flag in calls)

    def test_show_critical_difference(self):
        scores = pd.Series([0.2, 0.5, 0.55, 0.9], index=list('abcd'))
        rk.show_critical_difference(scores, [(1, 2), (0, 1)], xlabel='score')
