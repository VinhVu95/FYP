import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans

from time import time


def kmeans_on_pca_reduce(data, *num_clusters, use_pca=False):
    """
    perform k means clustering algorithm on principal component or original attributes
    Args:
        data: Panda Dataframe or numpy array
            original dataset (without target attributes)
        *num_clusters: different cluster instance
        use_pca: indicate if use PCA technique

    Returns: pca reduced (or original) data table, k means estimator

    """
    if use_pca is True:
        reduced_data = PCA(n_components=3).fit_transform(data)
    km_estimators = []
    for n in num_clusters:
        kmeans = KMeans(init='k-means++', n_clusters=n, n_init=10)
        kmeans.fit(reduced_data)
        km_estimators.append(kmeans)
    return reduced_data, km_estimators


def set_label(ax, label):
    if label == None:
        ax.set_xlabel('pca component 1')
        ax.set_ylabel('pca component 2')
        ax.set_zlabel('pca component 3')
    else:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])


def visualise_cluster3d(data, *num_clusters, label=None, use_pca=True):
    """
    Visualize the clustering of data (internal pattern and trends) using K Means algorithm

    Args:
        data: Panda Dataframe or numpy array
            original dataset (without target attributes).
        label:  list of string
            3D label axis
        use_pca: boolean
            whether use Principal Component Analysis Algorithm to reduce data dimension
        *num_clusters: running different cluster instance

    Returns: scatter plot of data

    """
    _data, km_estimators = kmeans_on_pca_reduce(data, use_pca=use_pca, *num_clusters)
    fignum = 1
    for est in km_estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, 1.6, 1.8], elev=48, azim=134)

        plt.cla()
        est.fit(_data)
        labels = est.labels_

        ax.scatter(_data[:, 0], _data[:, 1], _data[:, 2], c=labels.astype(np.float))
        set_label(ax, label)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        fignum = fignum + 1
    plt.show()


def bench_k_means(estimator, name, data, labels):
    """
    Bench mark k means
    Args:
        estimator: k means estimator
        name: PCA-based
        data: input data
        labels: target label (if provided)

    Returns: None

    """
    sample_size=300
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


