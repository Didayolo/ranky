#################################
######## VISUALIZATIONS #########
#################################

import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
import ranky as rk
from sklearn.manifold import TSNE, MDS
from mpl_toolkits.mplot3d import Axes3D

# critical difference does not work when this is enabled
#sns.set_theme(style = "darkgrid")

def autolabel(rects, values, round=2):
    """ Function used by `rk.show` to annotate bar plots.
    """
    values = np.round(values, round)
    for idx,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                values[idx],
                ha='center', va='bottom', rotation=0)

def show(m, rotation=90, title=None, size=2, annot=False,
         ylabel=None, xlabel=None, round=2, color='royalblue', cmap=None):
    """ Display a ranking or a prefrence matrix.

    If m is 1D: show ballot (bar plot).
    If m is 2D: show preferences (heatmap).

    TODO: annot argument adding the values in the plot.

    Args:
        rotation: x labels rotation.
        title: string - title of the figure.
        size: integer - higher value for a smaller figure.
        annot: If True, write the values.
        ylabel: string - y axis label.
        xlabel: string - x axis label.
        round: Number of decimals to display if annot is True.
        color: Color for 1D bar plot.
        cmap: Color map for 2D heatmap.
    """
    if isinstance(m, list): # convert to np.ndarray if needed
        m = np.array(m)
    dim = len(m.shape)
    if dim == 1: # 1D
        x = np.arange(len(m))
        bar_plot = plt.bar(x, m, align='center', color=color)
        if annot:
            autolabel(bar_plot, m, round=round)
        if rk.is_series(m):
            plt.xticks(x, m.index, rotation=rotation)
    elif dim == 2: # 2D
        fig, ax = plt.subplots(figsize=(m.shape[1]/size, m.shape[0]/size))
        sns.heatmap(m, ax=ax, annot=annot, linewidths=.2, fmt='0.'+str(round), cmap=cmap)
        x = np.arange(m.shape[1])
        if rk.is_dataframe(m):
            plt.xticks(x, m.columns, rotation=rotation)
    else:
        raise(Exception('Passed array must have only 1 or 2 dimension, not {}.'.format(dim)))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
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
        #if legend:
        #    plt.legend(colors, loc=legend_loc)
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
        **kwargs: arguments for rk.scatterplot function (e.g. fontsize).
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
        **kwargs: arguments for rk.scatterplot function (e.g. fontsize).
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

def overlaps(pos, couples):
    """ Used by critical difference.

    Checks if the horizontal line overlaps any existing horizontal line.
    """
    i, j = pos
    for i1, j1 in couples:
        if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
            return True
    return False

def merge_couples(couples):
    # Used by critical difference
    longest = [(i, j) for i, j in couples if not overlaps((i, j), couples)]
    return longest

def critical_difference(m, comparison_func=None, axis=1, **kwargs):
    """ Computes and draws a critical difference diagram.

    The goal of critical difference diagrams is to show the average scores of
    different candidates, and to group if their performance are not significantly
    different (using pairwise statistical tests).
    This function uses a comparison function (rk.p_wins by default).
    A comparison function f(a, b) should return True if a is significantly better than b.

    Args:
        m: Score matrix, array-like (use pd.DataFrame to name the candidates).
        comparison_func: Assymetrical function used to compare two candidates.
        The function comparison_func(a, b) should return 1 if a beats b and 0 otherwise.
        By default it's p_wins (defined in the same module), performing a binomial test.
        axis: Axis of judges.
        kwargs: Arguments for the comparison_func function.
    """
    m = pd.DataFrame(m) # casting if necessary
    scores = rk.score(m, axis=axis).sort_values()
    if axis == 0:
        m = m.T # if the candidates are in column, transpose the matrix
    couples = []
    for i in range(len(scores) - 1):
        for j in range(1, len(scores)):
            if i < j:
                _i, _j = scores.index[i], scores.index[j] # do not confuse indices in couples and in scores
                a, b = m.loc[_i], m.loc[_j]
                if rk.duel.declare_ties(a, b, comparison_func=comparison_func):
                    couples.append((i, j))
    show_critical_difference(scores, couples)

def show_critical_difference(scores, couples, arrow_vgap=.2, link_voffset=.15, link_vgap=.1, xlabel=None):
    """ Draws a critical difference diagram.

    The goal of critical difference diagrams is to show the average scores of
    different candidates, and to group if their performance are not significantly
    different (using pairwise statistical tests).

    Forked from https://github.com/mbatchkarov/critical_difference

    Critical difference diagrams can be seen in the following publications:
    - Janez Demsar, Statistical Comparisons of Classifiers over Multiple Data Sets, 7(Jan):1--30, 2006.
    - H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, P. Muller, Deep learning for time series classification: a review, Data Mining and Knowledge Discovery, 2018.

    Args:
        scores: List of average methods' scores, array-like. If scores is a pd.Series, the index will be used as names.
        couples: list of tuples representing the equivalence between neighbors (once sorted) e.g. [(0, 1), (1, 2), (4, 5)], based on indices in the array scores.
        arrow_vgap: vertical space between the arrows that point to method names, between 0 and 1.
        link_vgap: vertical space between the lines that connect methods that are not significantly different. Scale is 0 to 1, fraction of axis size
        link_voffset: offset from the axis of the links that connect non-significant methods
    """
    size = len(scores)
    names = list(range(size)) # default names: [0, 1, ...]
    if isinstance(scores, pd.Series):
        names = scores.index
    scores, names = (list(t) for t in zip(*sorted(zip(scores, names))))
    for pair in couples:
        assert all(0 <= idx < size for idx in pair), 'Check indices'
    # remove axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), subplot_kw=dict(frameon=False))
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().set_visible(False)
    y = [0] * size
    ax.plot(scores, y, 'ko')
    plt.xlim(0.9 * scores[0], 1.1 * scores[-1])
    plt.ylim(0, 1)
    # draw the x axis again
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    if xlabel: # add an optional label to the x axis
        ax.annotate(xlabel, xy=(xmax, 0), xytext=(0.95, 0.1), textcoords='axes fraction',
                                ha='center', va='center', fontsize=9)  # text slightly smaller
    half = int(ceil(size / 2.))
    # make sure the topmost annotation in at 90% of figure height
    ycoords = list(reversed([0.9 - arrow_vgap * i for i in range(half)]))
    ycoords.extend(reversed(ycoords))
    for i in range(size):
        ax.annotate(str(names[i]),
                    xy=(scores[i], y[i]),
                    xytext=(-.05 if i < half else .95, ycoords[i]),
                    textcoords='axes fraction', ha='center', va='center',
                    arrowprops={'arrowstyle': '-', 'connectionstyle': 'angle,angleA=0,angleB=90'})
    # draw horizontal lines linking non-significant methods
    linked_methods = merge_couples(couples)
    # where do the existing lines begin and end, (X, Y) coords
    used_endpoints = set()
    y = link_voffset
    dy = link_vgap
    # draw lines
    for i, (x1, x2) in enumerate(sorted(linked_methods)):
        if y > link_voffset and overlaps((x1, y - dy), used_endpoints):
            y -= dy
        elif overlaps((x1, y), used_endpoints):
            y += dy
        plt.hlines(y, scores[x1], scores[x2], linewidth=3)  # y, x0, x1
        used_endpoints.add((x1, y))
        used_endpoints.add((x2, y))
    plt.show()
