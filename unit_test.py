#####################
##### UNIT TEST #####
#####################

import unittest
import numpy as np
import pandas as pd
#from ranky import *
import ranky as rk

def test_distance_matrix():
    print('Distance matrix...')
    m_template = pd.read_csv('data/matrix.csv')
    m_template.index = m_template['index']
    m_template = m_template.drop('index', axis=1).rename_axis(None, axis = 0)
    print('Default')
    dist_matrix = rk.distance_matrix(m_template)
    print(dist_matrix)
    print('Levenshtein')
    dist_matrix = rk.distance_matrix(m_template, method='levenshtein')
    print(dist_matrix)

def test_generator():
    print('Testing generator...')
    G = rk.Generator()
    R = [5, 4, 3, 2, 1]
    G.fit(R)
    print('Judgement matrix')
    m = G.sample(n=9)
    print(m)
    print('Score ranking')
    r = rk.score(m)
    print(r)
    distance = rk.dist(rk.rank(R), rk.rank(r))
    correlation = rk.corr(R, r)
    print('hamming distance: {}'.format(distance))
    print('kendall tau correlation: {}'.format(correlation))
    print('Uninominal ranking')
    r = rk.uninominal(m)
    print(r)
    distance = rk.dist(rk.rank(R), rk.rank(r))
    correlation = rk.corr(R, r)
    print('hamming distance: {}'.format(distance))
    print('kendall tau correlation: {}'.format(correlation))

def test_metric():
    """ This simply test if the metrics got computed withtout error.
        It is not an unit testing.
    """
    print('Testing metrics...')
    y_true = pd.read_csv('data/test_metric/task.solution', sep=' ', header=None)
    y_pred = pd.read_csv('data/test_metric/task.predict', sep=' ', header=None)
    y_pred_proba = pd.read_csv('data/test_metric/task_proba.predict', sep=' ', header=None)
    for m in ['accuracy', 'balanced_accuracy', 'precision', 'average_precision', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse']:
        try:
            print('{}: {}'.format(m, rk.metric(y_true, y_pred, method=m)))
            print('{}: {}'.format(m, rk.metric(y_true, y_pred_proba, method=m)))
        except Exception as e:
            print('Failed for {}'.format(m))
            print(e)
    print('Combined loss (SAR by default)')
    print('{}'.format(rk.combined_metric(y_true, y_pred)))
    print('{}'.format(rk.combined_metric(y_true, y_pred_proba)))

def test_utilities():
    print('Testing utilities...')
    leaderboard = rk.read_codalab_csv('data/chems.csv')
    print(leaderboard.head())

class Test(unittest.TestCase):
    M = np.array([[0.3, 0.4, 0.6], [0.8, 0.8, 0.8], [0.1, 0.5, 0.7], [0.2, 0.2, 0.2], [0, 0, 0]])
    def test_rank(self):
        rank_M = np.array([[2., 3., 3.], [1., 1., 1.], [4., 2., 2.], [3., 4., 4.], [5., 5., 5.]])
        np.testing.assert_array_equal(rk.rank(self.__class__.M), rank_M)
    def test_borda(self):
        np.testing.assert_array_almost_equal(rk.borda(self.__class__.M), np.array([2.66666667, 1., 2.66666667, 3.66666667, 5.]))
    def test_majority(self):
        np.testing.assert_array_equal(rk.majority(self.__class__.M), np.array([0.4, 0.8, 0.5, 0.2, 0.]))
    def test_pairwise(self):
        np.testing.assert_array_equal(rk.pairwise(self.__class__.M), np.array([2., 4., 3., 1., 0.]))
    def test_pairwise2(self):
        np.testing.assert_array_equal(rk.pairwise(self.__class__.M, wins=rk.p_wins, pval=0.2), np.array([0., 0., 0., 0., 0.]))
    def test_consensus(self):
        rank_M = np.array([[1., 2., 3.], [1., 3., 2.], [1., 2., 3.]])
        np.testing.assert_array_equal(rk.consensus(rank_M, axis=1), np.array([True, False, False]))
    def test_winner_distance(self):
        self.assertEqual(rk.dist([1, 2, 3], [2, 1, 3], method='winner'), 1)
    def test_optimal_spearman_is_borda(self):
        """ Check that Borda count and Spearman optimal rank aggregation returns the same output on the template matrix.
        """
        m_template = pd.read_csv('data/matrix.csv')
        m_template.index = m_template['index']
        m_template = m_template.drop('index', axis=1).rename_axis(None, axis = 0)
        borda_rank = rk.rank(rk.borda(m_template), reverse=True)
        optimal_spearman_rank = rk.rank(rk.center(m_template, method='spearman'))
        print(borda_rank)
        print(optimal_spearman_rank)
        np.testing.assert_array_equal(borda_rank, optimal_spearman_rank)
    def test_kendall_w(self):
        M2 = np.array([[1, 2.5, 2.5, 4], [1, 2.5, 2.5, 4], [1, 2.5, 2.5, 4]])
        M3 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        self.assertEqual(rk.kendall_w(M2), 0.9)
        self.assertEqual(rk.kendall_w(M2, ties=True), 1.0)
        self.assertEqual(rk.kendall_w(M3), 0.0)
    def test_bayes_wins(self):
        a = [0, 0, 0.2, 0, 0.3]
        b = [1, 0.8, 1, 0.2, 0.2]
        bw = rk.bayes_wins(a, b)
        self.assertEqual(bw, False)
    def test_relative_difference(self):
        rd = rk.relative_difference([0, 0, 1], [0, 0, 1])
        self.assertEqual(rd, 0)
        rd = rk.relative_difference([0, 0], [0, 0])
        self.assertEqual(rd, 0)
        rd = rk.relative_difference([0.8, 0.1, 0.8], [0.2, 0.1, 0.2])
        self.assertAlmostEqual(rd, 0.4)
    def test_winner_distance(self):
        d = rk.winner_distance([1, 0.7, 0.2, 0.1, 0.1], [0.7, 1, 0.5, 0.4, 0.1])
        self.assertEqual(d, 0.25)
    def test_kendall_tau_distance(self):
        d = rk.kendall_tau_distance([0, 1, 2], [1, 2, 0])
        self.assertEqual(d, 2)
        d = rk.kendall_tau_distance([0, 1, 2], [0, 1, 2])
        self.assertEqual(d, 0)

if __name__ == '__main__':
    print('Compute various measures...')
    m = np.array([[1, 2, 3, 4], [1, 2, 4, 3], [1, 2, 4, 3], [1, 3, 2, 4], [2, 1, 3, 4], [1, 4, 3, 2]])
    #print(evolution_strategy(m, axis=1, l=5))
    print('Matrix:\n{}'.format(m))
    print('Concordance: {}'.format(rk.concordance(m, axis=0)))
    print('Kendall W: {}'.format(rk.kendall_w(m, axis=0)))
    print('Concordance (axis=1): {}'.format(rk.concordance(m, axis=1)))
    print('Kendall W (axis=1): {}'.format(rk.kendall_w(m, axis=1)))
    print('Euclidean center method: {}'.format(rk.center(m, method='euclidean', axis=0)))
    print('Pearson center method: {}'.format(rk.center(m, method='pearson', axis=0)))

    test_generator()
    test_metric()
    test_utilities()
    test_distance_matrix()

    print('Unit testing...')
    unittest.main()


'''
TODO to_binary
>>> m = np.array([[0.2, 0.3, 0.1], [0.5, 0.6, 0.7]])
>>> m
array([[0.2, 0.3, 0.1],
       [0.5, 0.6, 0.7]])
>>> rk.to_binary(m)
array([[0, 0, 0],
       [0, 1, 1]])
>>> rk.to_binary(m, unilabel=True)
array([[0, 1, 0],
       [0, 0, 1]])
>>> rk.to_binary(m, unilabel=True, at_least_one_class=True)
array([[0, 1, 0],
       [0, 0, 1]])
>>> rk.to_binary(m, unilabel=False, at_least_one_class=True)
array([[0, 1, 0],
       [0, 1, 1]])
>>> m
array([[0.2, 0.3, 0.1],
       [0.5, 0.6, 0.7]])
>>> rk.to_binary(m, unilabel=False, at_least_one_class=False)
array([[0, 0, 0],
       [0, 1, 1]])
'''

# TODO: tests for all functions
