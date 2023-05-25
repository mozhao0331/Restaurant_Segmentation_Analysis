import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

colors = ['xkcd:azure', 'yellowgreen', 'tomato', 'teal', 'indigo', 'aqua', 'orangered', 'orchid', 'black', 'xkcd:turquoise', 'xkcd:violet', 'aquamarine', 'chocolate', 'darkgreen', 'sienna', 'pink', 'lightblue', 'yellow', 'lavender', 'wheat', 'linen']


def discrete_scatter(x1, x2, y=None, markers=None, s=8, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=0.6, 
                     label_points=False, x1_annot=-0.1, x2_annot=0.1):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.
    Parameters
    ----------
    x1 : nd-array
        input data, first axis
    x2 : nd-array
        input data, second axis
    y : nd-array
        input data, discrete labels
    cmap : colormap
        Colormap to use.
    markers : list of string
        List of markers to use, or None (which defaults to 'o').
    s : int or float
        Size of the marker
    padding : float
        Fraction of the dataset range to use for padding the axes.
    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))        

    # unique_y = np.unique(y)
    unique_y, inds = np.unique(y, return_index=True)    

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []


    if len(unique_y) == 1: 
        cr = [-1]
    else: 
        cr = sorted([y[index] for index in sorted(inds)])

    if c is not None and len(c) == 1: 
        cr = c
    
    for (i, (yy, color_ind)) in enumerate(zip(unique_y, cr)):
        mask = y == yy
        # print(f'color_ind= {color_ind} and i = {i}')
        # if c is none, use color cycle
        color = colors[color_ind]
        # print('color: ', color)
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .2:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,                             
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])
    if label_points: 
        labs = [str(label) for label in list(range(0,len(x1)))]
        for i, txt in enumerate(labs):
            font_size=10
            ax.annotate(txt, (x1[i], x2[i]), xytext= (x1[i]+x1_annot, x2[i]+x2_annot), c='k', size = font_size)

    return lines    


def plot_dendrogram_clusters(
    df, 
    linkage_array, 
    hier_labels, 
    p=4,
    axis_0="osm_highway_exits_count_1mi",
    axis_1="dmm_gla_1mi",
    linkage_type='single', 
    title=None
): 
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    X = df.to_numpy()
    label_list = df.index
    dendrogram(linkage_array, ax=ax[0], p=p, labels=label_list, truncate_mode="level")
    ax[0].set_xlabel("Sample index")
    ax[0].set_ylabel("Cluster distance");
    ax[0].set_title(f"{linkage_type} linkage")
    
    index_0 = df.columns.get_loc(axis_0)
    index_1 = df.columns.get_loc(axis_1)

    discrete_scatter(X[:, index_0], X[:,index_1], hier_labels, markers='o', label_points=False, ax=ax[1]);
    ax[1].set_title(title)


def plot_original_clustered(X, model, labels):
    k = np.unique(labels).shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))    
    ax[0].set_title("Original dataset")
    ax[0].set_xlabel("Feature 0")
    ax[0].set_ylabel("Feature 1")    
    discrete_scatter(X[:, 0], X[:, 1], ax=ax[0]);
    # cluster the data into three clusters
    # plot the cluster assignments and cluster centers
    ax[1].set_title(f"{type(model).__name__} clusters")    
    ax[1].set_xlabel("Feature 0")
    ax[1].set_ylabel("Feature 1")

    discrete_scatter(X[:, 0], X[:, 1], labels, c=labels, markers='o', ax=ax[1]); 
    if type(model).__name__ == "KMeans": 
        discrete_scatter(
            model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], y=np.arange(0,k), s=15, 
            markers='*', markeredgewidth=1.0, ax=ax[1])

def pca_100_plot(train_data):
    pca_100 = PCA(n_components=100, whiten=True, random_state=42)
    pca_100.fit(train_data)
    cum_variance = np.cumsum(pca_100.explained_variance_ratio_)
    for i in range(0, len(cum_variance)):
        if cum_variance[i] > 0.9:
            num_comp = i + 1
            break
    explained_var_df = pd.DataFrame(
        data=cum_variance,
        columns=["cummulative variance_explained (%)"],
        index=range(1, 101),
    )
    explained_var_df.index.name = "n_components"
    plt.figure(figsize=(8, 6))
    plt.xticks(range(1, 101, 5))
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance ratio")
    plt.plot(range(1, 101), cum_variance)
    plt.grid()
    plt.show()
    return num_comp
