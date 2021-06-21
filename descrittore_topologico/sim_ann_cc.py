# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 2016

@author: Nicola Bevilacqua & Cristina Campi
"""
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from scipy.spatial.distance import cdist
from sklearn.cluster.k_means_ import _init_centroids

def _distance_data_centers(X, centers, distance):
    # Calculate the distances between X (data) and centers
    n_samples = X.shape[0]
    d = np.zeros(shape=(n_samples,), dtype=np.float64)
    d = cdist(X, centers, distance)
    d = np.fmax(d, np.finfo(np.float64).eps)

    return d

def _memberships_update_probabilistic(d, m):
    u = d ** (- 2. / (m - 1))
    u /= np.sum(u, axis=1)[:, np.newaxis]
    u = np.fmax(u, np.finfo(np.float64).eps)

    return u

def _init_memberships(X, centers, distance):
    d = _distance_data_centers(X, centers, distance)

    # fuzzy-probabilistic-style membership initialization with fixed fuzzyfier
    # parameter m=2
    u = _memberships_update_probabilistic(d, 2)

    return u, d


def _cost_function_Kmeans(X, centers,distance):
    # compute the cost function for the Kmeans

    '''
    cost = 0

    for i in range(X.shape[0]):

        distance = []

        for j in range(centers.shape[0]):
            distance.append((np.linalg.norm(X[i, :] - centers[j, :])) ** 2)

        cost += np.array(distance).min()

    #cost = np.sum(_distance_data_centers(X, centers, distance).min(axis=1))
    '''
    D = _distance_data_centers(X, centers, distance)

    cost = ((D.min(axis = 1)) ** 2).sum()
    return cost


def _cost_function_FKmeans(X, D, U, m):
    # compute te cost function for the FKmeans
    cost = ((D ** 2) * (U ** m)).sum()

    return cost

def f_k_means_sim_ann(X,constraint, n_clusters, init, T,dt,
                       tol_deltaE,  max_iter, distance, m):
    # if the initialization method is not 'k-means++',
    # an array of centroids is passed
    # and it is converted in float type
    if hasattr(init, '__array__'):
        n_clusters = init.shape[0]
        init = np.asarray(init, dtype=np.float64)

    # initialization deltaE
    deltaE = 10*tol_deltaE
    # initialization centers
    centers_conf = init
    # initialization of cost function and other stuff
    if constraint =='Kmeans':
        E_conf = _cost_function_Kmeans(X, centers_conf,distance)
    elif constraint =='FKmeans':
        d_conf = _distance_data_centers(X, centers_conf,distance)
        u_conf = _memberships_update_probabilistic(d_conf, m)
        E_conf = _cost_function_FKmeans(X, d_conf, u_conf, m)


    prob = []
    cont_temp = 0
    cont_conf = []
    cont_taken = []
    cont_lost = []
    config1 = 0
    config2 = 0

    # creation of the range of the data
    Fov1 = X.min(axis=0)
    Fov2 = X.max(axis=0)


    while deltaE > tol_deltaE:
        cont_conf.append(0)
        cont_taken.append(0)
        cont_lost.append(0)

        while cont_conf[cont_temp] < max_iter:
            centers_pert = np.empty((n_clusters,X.shape[1]))
            cont_conf[cont_temp] += 1
            E_pert = 0
            # perturbation
            for j in range(centers_conf.shape[1]):
                centers_pert[:, j] = centers_conf[:, j] + np.random.uniform(-(Fov2[j] - Fov1[j]) / 100,
                                                                    (Fov2[j] - Fov1[j]) / 100, 2)
            # check if still in the range of the data
            for i in range(centers_conf.shape[0]):
                centers_pert[i, j] = np.array([centers_pert[i, j], Fov1[j]]).max()
                centers_pert[i, j] = np.array([centers_pert[i, j], Fov2[j]]).min()

            # computation of cost function and other stuff
            if constraint == 'Kmeans':
                E_pert = _cost_function_Kmeans(X, centers_pert, distance)
            elif constraint == 'FKmeans':
                d_pert = _distance_data_centers(X, centers_pert,distance)
                u_pert = _memberships_update_probabilistic(d_pert, m)
                E_pert = _cost_function_FKmeans(X, d_pert, u_pert, m)

            # check on cost function
            if E_conf > E_pert:
                E_conf = E_pert
                centers_conf = centers_pert
                cont_taken[cont_temp] += 1
                if constraint == 'FKmeans':
                    d_conf = d_pert
                    u_conf = u_pert
            else:
                p = np.array([1, np.exp(-(E_pert - E_conf) / T)]).min()
                prob.append(p)
                eps = np.random.uniform(0, 1)

                # accept the update
                if eps < p:
                    E_conf = E_pert
                    centers_conf = centers_pert
                    cont_taken[cont_temp] += 1
                    if constraint == 'FKmeans':
                        d_conf = d_pert
                        u_conf = u_pert
                else:
                    cont_lost[cont_temp] += 1

        if cont_temp % 2 == 0:
            config1 = E_conf
        else:
            config2 = E_conf


        deltaE = np.linalg.norm(config1 - config2)
        T *= dt
        cont_temp += 1

    if constraint == 'Kmeans':
        predicted_labels = np.argmax(_distance_data_centers(X, centers_conf, distance), axis=1)
    elif constraint == 'FKmeans':
        predicted_labels = np.argmax(u_conf, axis=1)


    return centers_conf, predicted_labels


class SimAnnKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, constraint = 'Kmeans',n_clusters=2, init='k-means++', T = 15, dt = 0.9,
                 tol_deltaE=1e-6,max_iter=200,distance = 'euclidean',m=2.0):

        self.constraint = constraint
        self.n_clusters = n_clusters
        self.init = init
        self.T = T
        self.dt = dt
        self.tol_deltaE=tol_deltaE
        self.max_iter = max_iter
        self.distance = distance
        self.m = m

    def fit(self, X, y=None):
        if y is None:
            self.centers = _init_centroids(X,
                                      self.n_clusters,
                                      self.init,
                                      x_squared_norms=row_norms(X,
                                                                squared=True))
        else:
            n_labels = int(np.max(y))
            self.centers = np.zeros([n_labels + 1, np.shape(X)[1]])
            for l in np.arange(n_labels + 1):
                self.centers[l, :] = np.mean(X[np.where(y==l)[0],:],axis=0)

        cluster_centers, predicted_labels = \
            f_k_means_sim_ann(X,
                      constraint=self.constraint,
                      n_clusters=self.n_clusters,
                      init=self.centers,
                      T=self.T,
                      dt=self.dt,
                      tol_deltaE=self.tol_deltaE,
                      max_iter=self.max_iter,
                      distance=self.distance,
                      m=self.m)

        self.labels_ = predicted_labels
        self.cluster_centers_ = cluster_centers

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
         X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        d = _distance_data_centers(X, self.cluster_centers_, self.distance)
        predicted_labels = np.argmin(d, axis=1)

        return predicted_labels




class SimAnnFKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, constraint = 'FKmeans',n_clusters=2, init='k-means++', T = 15, dt = 0.9,
                 tol_deltaE=1e-6,max_iter=200,distance = 'euclidean',m=2.0):

        self.constraint = constraint
        self.n_clusters = n_clusters
        self.init = init
        self.T = T
        self.dt = dt
        self.tol_deltaE=tol_deltaE
        self.max_iter = max_iter
        self.distance = distance
        self.m = m

    def fit(self, X, y=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """

        # usare le label per calcolare i centroidi e poi passarli come
        # inizializzazione per il predict
        if y is None:
            self.centers = _init_centroids(X,
                                   self.n_clusters,
                                   self.init,
                                   x_squared_norms=row_norms(X,
                                                             squared=True))
        else:
            n_labels = int(np.max(y))
            self.centers = np.zeros([n_labels + 1, np.shape(X)[1]])
            for l in np.arange(n_labels + 1):
                self.centers[l, :] = np.mean(X[np.where(y==l)[0],:],axis=0)




        cluster_centers, predicted_labels = \
            f_k_means_sim_ann(X,
                      constraint=self.constraint,
                      n_clusters=self.n_clusters,
                      init=self.centers,
                      T=self.T,
                      dt=self.dt,
                      tol_deltaE=self.tol_deltaE,
                      max_iter=self.max_iter,
                      distance=self.distance,
                      m=self.m)

        self.labels_ = predicted_labels
        self.cluster_centers_ = cluster_centers

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
         X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        d = _distance_data_centers(X, self.cluster_centers_, self.distance)
        predicted_labels = np.argmin(d, axis=1)

        return predicted_labels
