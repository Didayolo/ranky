#################################
######## VISUALIZATIONS #########
#################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import ranky.ranking as rk
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

sns.set(style = "darkgrid")

def show(m, rotation=90, title=None, size=2):
    """ If m is 1D: show ballot (bar plot).
        If m is 2D: show preferences (heatmap).

        :param rotation: x labels rotation.
        :param title: string - title of the figure.
        :param size: integer - higher value for a smaller figure.
    """
    dim = len(m.shape)
    if dim == 1: # 1D
        x = np.arange(len(m))
        plt.bar(x, m, align='center')
        if rk.is_series(m):
            plt.xticks(x, m.index, rotation=rotation)
    elif dim == 2: # 2D
        fig, ax = plt.subplots(figsize=(m.shape[1]/size, m.shape[0]/size))
        sns.heatmap(m, ax=ax)
        x = np.arange(m.shape[1])
        if rk.is_dataframe(m):
            plt.xticks(x, m.columns, rotation=rotation)
    else:
        raise(Exception('Passed array must have only 1 or 2 dimension, not {}.'.format(dim)))
    if title is not None:
        plt.title(title)
    plt.show()

def show_learning_curve(h):
    """ Display learning curve.

        :param h: list representing the history of scores.
    """
    plt.plot(range(len(h)), h)
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.show()

def show_graph(matrix, names=None):
    """ Show a directed graph represented by a binary matrix.

        :param matrix: binary matrix. matrix[i, j] = 1 indicates an edge from i to j.
        :param names: list representing the names of the vertices.
    """
    G = nx.DiGraph()
    n = len(matrix)
    nodes = range(n)
    if names is not None:
        nodes = names
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                G.add_edge(nodes[i], nodes[j])
    nx.draw_circular(G, with_labels=True, node_size=2500, font_size=8, font_weight='bold')
    plt.show()

def tsne(m, axis=0, dim=2):
    """ Use T-SNE algorithm to show the matrix m in a 2 or 3 dimensions space.

        :param axis: axis of dimensionality reduction.
        :param dim: number of dimensions. 2 for 2D plot, 3 for 3D plot.
    """
    names = None
    if axis == 0:
        if rk.is_dataframe(m):
            names = m.columns
        m = m.T # transpose
    elif axis == 1:
        if rk.is_dataframe(m):
            names = m.index
    else:
        raise Excpetion('axis must be 0 or 1.')
    m_transformed = TSNE(n_components=dim).fit_transform(m)
    if dim == 2: # 2 dimensions
        x, y = [m_transformed[:, i] for i in range(m_transformed.shape[1])] # take columns
        sns.scatterplot(x, y, hue=names)
        plt.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    elif dim == 3: # 3 dimensions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        x, y, z = [m_transformed[:, i] for i in range(m_transformed.shape[1])] # take columns
        ax.scatter(x, y, z)
        plt.show()
    else:
        raise Exception('dim must be 2 or 3.')


# Critical difference diagrams
# https://github.com/hfawaz/cd-diagram/blob/master/main.py
