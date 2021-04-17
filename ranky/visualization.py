#################################
######## VISUALIZATIONS #########
#################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import ranky as rk
from sklearn.manifold import TSNE, MDS
from mpl_toolkits.mplot3d import Axes3D

sns.set(style = "darkgrid")

def autolabel(rects, values, round=2):
    """ Function used by `rk.show` to annotate bar plots.
    """
    values = np.round(values, round)
    for idx,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                values[idx],
                ha='center', va='bottom', rotation=0)

def show(m, rotation=90, title=None, size=2, annot=False, round=2):
    """ Display a ranking or a prefrence matrix.

    If m is 1D: show ballot (bar plot).
    If m is 2D: show preferences (heatmap).

    TODO: annot argument adding the values in the plot.

    Args:
        rotation: x labels rotation.
        title: string - title of the figure.
        size: integer - higher value for a smaller figure.
        annot: If True, write the values.
        round: Number of decimals to display if annot is True.
    """
    dim = len(m.shape)
    if dim == 1: # 1D
        x = np.arange(len(m))
        bar_plot = plt.bar(x, m, align='center')
        if annot:
            autolabel(bar_plot, m, round=round)
        if rk.is_series(m):
            plt.xticks(x, m.index, rotation=rotation)
    elif dim == 2: # 2D
        fig, ax = plt.subplots(figsize=(m.shape[1]/size, m.shape[0]/size))
        sns.heatmap(m, ax=ax, annot=True, linewidths=.2, fmt='0.'+str(round))
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

    Args:
        h: list representing the history of scores.
    """
    plt.plot(range(len(h)), h)
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.show()

def show_graph(matrix, names=None):
    """ Show a directed graph represented by a binary matrix.

    Args:
        matrix: binary matrix. matrix[i, j] = 1 indicates an edge from i to j.
        names: list representing the names of the vertices.
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

def scatterplot(m, dim=2, names=None, colors=None, fontsize=8, pointsize=60, big_display=True, legend=False, legend_loc='best'):
    """ 2D or 3D scatterplot.

    Args:
        m: data
        dim: 2 or 3.
        names: vector of names to display on each point.
        colors: vector of numbers or categories of the size of the number of points.
                If None it will be replaced by names.
        fontsize: text font size (integer).
        pointsize: size of data points (integer).
        big_display: plot the figure in a big format if True.
        legend: if True, add legend of colors.
        legend_loc: location of legend. See matplotlib.pyplot.legend for details.
    """
    if colors is None:
        colors = names
    if dim == 2: # 2 dimensions
        x, y = [m[:, i] for i in range(m.shape[1])] # take columns
        scat = sns.scatterplot(x, y, hue=colors, s=pointsize, legend=(legend and 'brief'))
        if names is not None: # TEXT #
            for line in range(0, m.shape[0]):
                scat.text(x[line]+0.01, y[line], names[line], horizontalalignment='left',
                         fontsize=fontsize, color='black', weight='semibold')
        if legend:
            plt.legend(colors, loc=legend_loc)
    elif dim == 3: # 3 dimensions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        x, y, z = [m[:, i] for i in range(m.shape[1])] # take columns
        ax.scatter(x, y, z) #, c=range(len(names)))
    else:
        raise Exception('dim must be 2 or 3.')
    if big_display:
        fi = plt.gcf()
        fi.set_size_inches(12, 8) # change plot size
    plt.show()

def tsne(m, axis=0, dim=2, **kwargs):
    """ Use T-SNE algorithm to show the matrix m in a 2 or 3 dimensions space.

    Args:
        axis: axis of dimensionality reduction.
        dim: number of dimensions. 2 for 2D plot, 3 for 3D plot.
        **kwargs: arguments for rk.scatterplot function (e.g. fontsize, pointsize).
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
    # Display
    scatterplot(m_transformed, dim=dim, names=names, **kwargs)

def mds_from_dist_matrix(distance_matrix, dim=2, names=None, **kwargs):
    """ Multidimensional scaling plot from a symmetric distance matrix (pairwise distances).

    See: https://en.wikipedia.org/wiki/Multidimensional_scaling

    Args:
        m: distance matrix.
        dim: number of dimensions to plot (2 or 3).
        names: names of objects. Will be overwritten if distance_matrix is a pd.DataFrame.
        **kwargs: arguments for scatterplot function (e.g. fontsize).
    """
    if rk.is_dataframe(distance_matrix):
        names = distance_matrix.columns
    transformer = MDS(n_components=dim, dissimilarity='precomputed')
    m_transformed = transformer.fit_transform(distance_matrix)
    # Display
    scatterplot(m_transformed, dim=dim, names=names, **kwargs)

def mds(m, axis=0, dim=2, method='spearman', **kwargs):
    """ Multidimensional scaling plot from a preference matrix.

    See: https://en.wikipedia.org/wiki/Multidimensional_scaling

    Args:
        m: preference matrix.
        dim: number of dimensions to plot (2 or 3).
        method: any metric method.
        **kwargs: arguments for scatterplot function (e.g. fontsize).
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
    # Compute pairwise distances
    dist_matrix = rk.distance_matrix(m, method=method)
    # Call the plot functions
    mds_from_dist_matrix(dist_matrix, dim=dim, names=names, **kwargs)


# Critical difference diagrams
# https://github.com/hfawaz/cd-diagram/blob/master/main.py
