#################################
######## VISUALIZATIONS #########
#################################

import inspect
import itertools as it
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

# All plotting functions call plt.show() at the end.

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
    """ Display a ranking or a preference matrix.

    If m is 1D: show the ballot as a bar plot.
    If m is 2D: show the preferences as a heatmap.

    Args:
        m: 1D or 2D array-like. Use pd.Series or pd.DataFrame to display names.
        rotation: Rotation of the x labels.
        title: Title of the figure.
        size: Higher value for a smaller figure (2D only).
        annot: If True, write the values.
        ylabel: Label of the y axis.
        xlabel: Label of the x axis.
        round: Number of decimals to display if annot is True.
        color: Color of the bars (1D only).
        cmap: Color map of the heatmap (2D only).
    """
    if isinstance(m, list): # convert to np.ndarray if needed
        m = np.array(m)
    dim = len(m.shape)
    if dim == 1: # 1D
        fig, ax = plt.subplots()
        x = np.arange(len(m))
        bar_plot = plt.bar(x, m, align='center', color=color)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True) # set the lines below the bars
        if annot:
            autolabel(bar_plot, m, round=round)
        if rk.is_series(m):
            plt.xticks(x, m.index, rotation=rotation)
    elif dim == 2: # 2D
        fig, ax = plt.subplots(figsize=(m.shape[1]/size, m.shape[0]/size))
        sns.heatmap(m, ax=ax, annot=annot, linewidths=.2, fmt='0.'+str(round), cmap=cmap)
        x = np.arange(m.shape[1]) + 0.5 # heatmap cells are centered on x + 0.5
        if rk.is_dataframe(m):
            plt.xticks(x, m.columns, rotation=rotation)
    else:
        raise ValueError('Passed array must have only 1 or 2 dimension, not {}.'.format(dim))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def show_learning_curve(h):
    """ Display a learning curve (score as a function of epochs).

    Args:
        h: List of scores, one per epoch.
    """
    plt.plot(range(len(h)), h)
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.show()

def show_graph(matrix, names=None, with_labels=True, node_size=2500, font_size=8, font_weight='bold', arrowsize=10):
    """ Show a directed graph represented by a binary matrix.

    Can be used to display the graph returned by `rk.pairwise(m, return_graph=True)`.

    Args:
        matrix: Binary matrix. matrix[i, j] = 1 indicates an edge from i to j.
        names: Names of the vertices.
        with_labels: Display names if True.
        node_size: Size of the nodes.
        font_size: Size of the font to display names.
        font_weight: 'bold' for bold, else refer to networkx documentation.
        arrowsize: Size of the arrows.
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
    nx.draw_circular(G, with_labels=with_labels, node_size=node_size, font_size=font_size, font_weight=font_weight, arrowsize=arrowsize)
    plt.show()

def scatterplot(m, dim=2, names=None, colors=None, fontsize=8, pointsize=60, big_display=True, legend=False, legend_loc='best'):
    """ 2D or 3D scatterplot.

    Args:
        m: np.ndarray of shape (n_points, dim).
        dim: 2 or 3.
        names: Names to display next to each point.
        colors: Numbers or categories, one per point, used to color the points.
            If None, names are used (2D only).
        fontsize: Font size of the names.
        pointsize: Size of the points.
        big_display: If True, plot the figure in a big format.
        legend: If True, add a legend of the colors (2D only).
        legend_loc: Location of the legend. See matplotlib.pyplot.legend for details.
    """
    m = np.asarray(m)
    if colors is None:
        colors = names
    if dim == 2: # 2 dimensions
        x, y = m[:, 0], m[:, 1]
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=colors, s=pointsize, legend=(legend and 'brief'), ax=ax)
        if names is not None: # TEXT #
            for line in range(0, m.shape[0]):
                ax.text(x[line]+0.01, y[line], names[line], horizontalalignment='left',
                        fontsize=fontsize, color='black', weight='semibold')
        if legend and colors is not None:
            ax.legend(loc=legend_loc)
        # Put back old matplotlib grid
        ax.set_facecolor('#EAEAF2')
        ax.grid(True, color='white')
    elif dim == 3: # 3 dimensions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = m[:, 0], m[:, 1], m[:, 2]
        ax.scatter(x, y, z, s=pointsize)
        if names is not None:
            for line in range(0, m.shape[0]):
                ax.text(x[line], y[line], z[line], names[line], fontsize=fontsize)
    else:
        raise ValueError('dim must be 2 or 3.')
    if big_display:
        plt.gcf().set_size_inches(12, 8) # change plot size
    plt.show()

def tsne(m, axis=0, dim=2, perplexity=None, **kwargs):
    """ Plot the rows or columns of m in 2D or 3D, using t-SNE.

    Args:
        m: 2D matrix. Use a pd.DataFrame to display names.
        axis: 0 to plot the columns (e.g. judges), 1 to plot the rows (e.g. candidates).
        dim: Number of dimensions. 2 for 2D plot, 3 for 3D plot.
        perplexity: t-SNE perplexity. By default min(30, n_points - 1).
        **kwargs: Arguments for `rk.scatterplot` (e.g. fontsize, pointsize).
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
        raise ValueError('axis must be 0 or 1.')
    m = np.asarray(m)
    if perplexity is None:
        perplexity = min(30, m.shape[0] - 1) # must be lower than the number of points
    m_transformed = TSNE(n_components=dim, perplexity=perplexity).fit_transform(m)
    # Display
    scatterplot(m_transformed, dim=dim, names=names, **kwargs)

def mds_from_dist_matrix(distance_matrix, dim=2, names=None, **kwargs):
    """ Multidimensional scaling plot from a symmetric distance matrix (pairwise distances).

    See: https://en.wikipedia.org/wiki/Multidimensional_scaling

    Args:
        distance_matrix: Square matrix of distances.
        dim: Number of dimensions to plot (2 or 3).
        names: Names of the objects. Overwritten if distance_matrix is a pd.DataFrame.
        **kwargs: Arguments for `rk.scatterplot` (e.g. fontsize).
    """
    if rk.is_dataframe(distance_matrix):
        names = distance_matrix.columns
    if 'metric_mds' in inspect.signature(MDS).parameters: # scikit-learn >= 1.8
        transformer = MDS(n_components=dim, metric='precomputed', init='random')
    else:
        transformer = MDS(n_components=dim, dissimilarity='precomputed')
    m_transformed = transformer.fit_transform(distance_matrix)
    # Display
    scatterplot(m_transformed, dim=dim, names=names, **kwargs)

def mds(m, axis=0, dim=2, method='spearman', **kwargs):
    """ Multidimensional scaling plot from a preference matrix.

    Pairwise distances are computed with `method`, then the points are placed
    in 2D or 3D so that their distances are preserved as much as possible.
    Correlations are converted to distances with (1 - correlation).

    See: https://en.wikipedia.org/wiki/Multidimensional_scaling

    Args:
        m: Preference matrix. Use a pd.DataFrame to display names.
        axis: 0 to plot the columns (e.g. judges), 1 to plot the rows (e.g. candidates).
        dim: Number of dimensions to plot (2 or 3).
        method: Distance or correlation method (see `rk.any_metric`).
        **kwargs: Arguments for `rk.scatterplot` (e.g. fontsize).
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
        raise ValueError('axis must be 0 or 1.')
    # Compute pairwise distances
    dist_matrix = rk.distance_matrix(m, method=method)
    if method in rk.CORR_METHODS: # higher correlation means closer
        dist_matrix = 1 - dist_matrix
    # Call the plot functions
    mds_from_dist_matrix(dist_matrix, dim=dim, names=names, **kwargs)

def overlaps(pos, couples):
    """ Check if the horizontal line `pos` is contained in a line of `couples`.

    Used by `show_critical_difference`.
    """
    i, j = pos
    for i1, j1 in couples:
        if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
            return True
    return False

def merge_couples(couples):
    """ Keep only the couples that are not contained in a longer one.

    Used by `show_critical_difference`.
    """
    longest = [(i, j) for i, j in couples if not overlaps((i, j), couples)]
    return longest

def critical_difference(m, comparison_func=None, axis=1, **kwargs):
    """ Compute and draw a critical difference diagram.

    Critical difference diagrams show the average score of each candidate, and
    link the candidates whose performances are not significantly different
    (using pairwise statistical tests).

    Args:
        m: Score matrix, array-like (use pd.DataFrame to name the candidates).
        comparison_func: Asymmetrical function used to compare two candidates.
            comparison_func(a, b) should return 1 (or True) if a is significantly
            better than b and 0 otherwise. By default it is `rk.p_wins`, performing
            a binomial test. See the `rk.duel` module for other options.
        axis: Axis of judges.
        **kwargs: Arguments passed to comparison_func (e.g. pval for `rk.p_wins`).
    """
    m = pd.DataFrame(m) # casting if necessary
    scores = rk.score(m, axis=axis).sort_values()
    if axis == 0:
        m = m.T # if the candidates are in column, transpose the matrix
    couples = []
    for i, j in it.combinations(range(len(scores)), 2):
        _i, _j = scores.index[i], scores.index[j] # do not confuse indices in couples and in scores
        a, b = m.loc[_i], m.loc[_j]
        if rk.declare_ties(a, b, comparison_func=comparison_func, **kwargs):
            couples.append((i, j))
    show_critical_difference(scores, couples)

def show_critical_difference(scores, couples, arrow_vgap=.2, link_voffset=.15, link_vgap=.1, xlabel=None):
    """ Draw a critical difference diagram from precomputed scores and ties.

    See `critical_difference` to compute the ties automatically.

    Forked from https://github.com/mbatchkarov/critical_difference

    Critical difference diagrams can be seen in the following publications:
    - Janez Demsar, Statistical Comparisons of Classifiers over Multiple Data Sets, 7(Jan):1--30, 2006.
    - H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, P. Muller, Deep learning for time series classification: a review, Data Mining and Knowledge Discovery, 2018.

    Args:
        scores: Average score of each method, array-like. If scores is a pd.Series, its index is used as names.
        couples: List of tuples of indices (in the sorted scores) of methods that are not
            significantly different, e.g. [(0, 1), (1, 2), (4, 5)].
        arrow_vgap: Vertical space between the arrows that point to method names, between 0 and 1.
        link_voffset: Offset from the axis of the links that connect non-significant methods.
        link_vgap: Vertical space between the lines that connect methods that are not
            significantly different. Fraction of the axis size, between 0 and 1.
        xlabel: Optional label of the x axis.
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
