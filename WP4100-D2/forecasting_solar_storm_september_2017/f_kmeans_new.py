# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:34:24 2015

@author: Federico Benvenuto & Annalisa Perasso
"""

import warnings

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms


def _distance_data_centers(X, centers, distance):
    # Calculate the distances between X (data) and centers
    n_samples = X.shape[0]
    d = np.zeros(shape=(n_samples,), dtype=np.float64)
    d = cdist(X, centers, distance)
    d = np.fmax(d, np.finfo(np.float64).eps)

    return d


def _labels_computation(u):
    # Labels computation
    labels = np.argmax(u, axis=1)

    return labels


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix u. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def _no_empty_clusters(centers, labels):
    r = -1

    kk = centers.shape[0]
    KMIst = np.array([], dtype=int)

    for i in range(kk):
        KMIst = np.hstack((KMIst, np.sum(labels == i)))
    Nempty = sum(KMIst > 0)
    NKM = -1 * np.ones((labels.shape[0], 1), dtype=int)
    NC = np.zeros((Nempty, centers.shape[1]))
    for i in range(kk):
        if KMIst[i] > 0:
            r = r + 1
            NKM[labels == i] = r
            NC[r][:] = centers[i][:]

    return NKM, NC


def _init_memberships(X, centers, distance):
    d = _distance_data_centers(X, centers, distance)

    # fuzzy-probabilistic-style membership initialization with fixed fuzzyfier
    # parameter m=2
    u = _memberships_update_probabilistic(d, 2)

    return u, d


def _memberships_update_probabilistic(d, m):
    u = d ** (- 2. / (m - 1))
    u /= np.sum(u, axis=1)[:, np.newaxis]
    u = np.fmax(u, np.finfo(np.float64).eps)

    return u


def _memberships_update_possibilistic(d, m, eta):
#    u = (1 + ((d ** 2) / eta) ** (1. / (m - 1))) ** (-1)
#    u = np.fmax(u, np.finfo(np.float64).eps)

    u = np.exp(-((d ** 2) / eta))
    u = np.fmax(u, np.finfo(np.float64).eps)

    return u


def _centers_update(X, um):
    centers = np.dot(um.T, X)
    centers /= um.sum(axis=0)[:, np.newaxis]

    return centers


def _eta_update(um,u, d):
    mask = u > 0.3
    eta = ((mask * d) ** 2).sum(axis=0)
    eta /= mask.sum(axis=0)
    return eta

def _eta_compute(um,d):
    eta = ((um * d) ** 2).sum(axis=0)
    s = (um ** 2).sum(axis=0)
    eta /= s

    return eta


def _f_k_means_probabilistic(X, u_old, n_clusters, m, distance):
    #
    um = u_old ** m

    # Calculate cluster centers
    centers = _centers_update(X, um)

    # Calculate the distances between X (data) and centers
    d = _distance_data_centers(X, centers, distance)

    # Probabilistic cost function calculation
    jm = (um * d ** 2).sum()

    # Membership update
    u = _memberships_update_probabilistic(d, m)

    return centers, u, jm, d


def _f_k_means_possibilistic(X, u_old, n_clusters, m, distance, eta):
    um = u_old ** m

    # Calculate cluster centers
    centers = _centers_update(X, um)

    # Calculate the distances between X (data) and centers
    d = _distance_data_centers(X, centers, distance)

    # Possibilistic cost function calculation da controllare!
    #jm = (um * d ** 2).sum() + np.dot((1 - u_old) ** m, eta).sum()
    jm = (u_old * d**2).sum()+ np.dot(u_old*np.log(u_old) - u_old,eta).sum()
    # Membership update
    u = _memberships_update_possibilistic(d, m, eta)

    return centers, u, jm, d


def J_prob(z, *params):
    centers, u = z
    X, m, distance = params

    return (((u ** m) *
             _distance_data_centers(X, centers, distance) ** 2).sum())


def J2_poss(z, *params):
    centers, u = z
    X, m, eta, distance = params

    return (np.dot((1 - u) ** m, eta).sum())


def J_poss(z, *params):
    centers, u = z
    X, m, eta, distance = params

    return J_prob(z, *params) + J2_poss(z, *params)


def f_k_means_main_loop(X, n_clusters, m, u, centers, d, tol_memberships,
                        tol_centroids, max_iter, constraint, distance, eta):
    # Initialization loop parameters
    p = 0
    jm = np.empty(0)

    # Main fcmeans loop
    while p <  max_iter - 1:
        u_old = u.copy()
        centers_old = centers.copy()

        [centers, u, inertia, d] = \
            _f_k_means_probabilistic(X,
                                     u_old,
                                     n_clusters,
                                     m,
                                     distance)


        jm = np.hstack((jm, inertia))
        p += 1

        # Stopping rule on memberships
        if np.linalg.norm(u - u_old) < tol_memberships:
            print('Stopping rule on memberships')
            break

        # Stopping rule on centroids
        if np.linalg.norm(centers - centers_old) < tol_centroids:
            print('Stopping rule on centroids')
            break

    # Final calculations
    fpc = _fp_coeff(u)

    # Labels computation
    labels = _labels_computation(u)

    return centers, labels, inertia, p, u, fpc




def f_k_means(X, n_clusters, m, tol_memberships, tol_centroids, max_iter, init,
              constraint, distance, n_init, sigma, eta):
    # if the initialization method is not 'k-means++',
    # an array of centroids is passed
    # and it is converted in float type
    if hasattr(init, '__array__'):
        n_clusters = init.shape[0]
        init = np.asarray(init, dtype=np.float64)

    # Initialize centers and memberships
    n_samples, n_features = X.shape

    centers = _init_centroids(X,
                              n_clusters,
                              init,
                              random_state=True,
                              x_squared_norms=row_norms(X, squared=True))

    u, d = _init_memberships(X, centers, distance)
    labels = _labels_computation(u)
    # Choose the optimization method

    centers, labels, inertia, n_iter, u, fpc = \
        f_k_means_main_loop(X,
                            n_clusters,
                            m,
                            u,
                            centers,
                            d,
                            tol_memberships,
                            tol_centroids,
                            max_iter,
                            constraint,
                            distance,
                            eta)

    return centers, labels


def p_k_means(X, n_clusters, m, u, centers, d, tol_memberships = 1e-8,tol_centroids=1e-4,max_iter=100,distance='euclidean', sigma=0.5, eta=None):

    n_run = 1

    for it in np.arange(n_run):

        centers_aux, labels, inertia, n_iter, u, fpc, eta = \
            p_k_means_main_loop(X,
                            n_clusters,
                            m,
                            u,
                            centers,
                            d,
                            tol_memberships,
                            tol_centroids,
                            max_iter,
                            distance,
                            eta)

        if it==0:
            C_total = centers_aux
            labels_total  = labels
            eta_total = eta
            inertia_total = inertia
        else:
            if  inertia<inertia_total:
                C_total = centers_aux
                labels_total = labels
                eta_total = eta
                inertia_total = inertia
    C_total, labels_total,eta_total = check_anomali_pcm(X, C_total, labels_total, sigma, distance,m)

    d_training = _distance_data_centers(X, C_total, distance)
    u_training = _memberships_update_possibilistic(d_training, m, eta_total)
    labels_total = _labels_computation(u_training)

    return C_total, labels_total, eta_total

def p_k_means_main_loop(X, n_clusters, m, u, centers, d, tol_memberships,
                        tol_centroids, max_iter, distance, eta):
    # Initialization loop parameters
    p = 0
    jm = np.empty(0)
    aux_eta = 0
    if eta == None:
        eta = _eta_compute(u**m,d)
        aux_eta = 1
    else:
    	eta = eta*np.ones((n_clusters,))
        
    # Main fcmeans loop
    while p <  max_iter - 1:
        u_old = u.copy()
        centers_old = centers.copy()
        if p == 1 and aux_eta ==1:
            um = u_old ** m
            eta = _eta_update(um,u_old, d)
        [centers, u, inertia, d] = \
             _f_k_means_possibilistic(X,
                                     u_old,
                                     n_clusters,
                                     m,
                                     distance,
                                     eta)

        jm = np.hstack((jm, inertia))
        p += 1

        # Stopping rule on memberships
        if np.linalg.norm(u - u_old) < tol_memberships:
        # if p>1 and np.abs(jm[p-1]-jm[p-2])/jm[p-1]<tol_memberships:
            print('Stopping rule on memberships')
            break

        # Stopping rule on centroids
        if np.linalg.norm(centers - centers_old) < tol_centroids:
            print('Stopping rule on centroids')
            break

    # Final calculations
    fpc = _fp_coeff(u)

    # Labels computation
    labels = _labels_computation(u)

    return centers, labels, inertia, p, u, fpc, eta

'''
def check_anomali(X, centers, labels, sigma, distance,eta,m):
    n_clusters = centers.shape[0]
    if n_clusters > 2:
        DATA = X.copy()
        label_DATA = np.arange(0, DATA.shape[0], 1)
        C_total = np.empty((0, DATA.shape[1]))
        NC_mean = np.zeros((1, DATA.shape[1]))
        label_total = -1 * np.ones((DATA.shape[0], 1), dtype=int)
        count_cluster = -1  # deve partire da -1 - vedi riga sopra

        NKM, NC = _no_empty_clusters(centers, labels)
        DC = cdist(NC, NC, distance)


        if np.max(NKM)==0:
            aux_index = np.arange(0, 1, dtype=int)
        else:
            aux_index = np.arange(0, np.max(NKM), dtype=int)

        distmed = np.zeros((NC.shape[0], 2))
        Anomali = np.array([], dtype=int)
        card = np.zeros((NC.shape[0]))
        indiceAnomali = np.array([])
        for ii in range(NC.shape[0]):
            DCi = DC[ii][:]
            distmed[ii][0] = np.mean(DCi[DCi > 0.])
            distmed[ii][1] = np.std(DCi[DCi > 0.])

        AD = np.mean(distmed[:, 0])
        prod = np.divide(distmed[:, 0], distmed[:, 1])
        mprod = np.mean(prod)
        sprod = np.std(prod)

        if np.max(distmed[:, 1]) != 0.:
            for ii in range(NC.shape[0]):
                if distmed[ii, 0] > AD + sigma * distmed[ii, 1]:
                    Anomali = np.hstack((Anomali, ii))
                elif prod[ii] > mprod + sigma * sprod:
                    Anomali = np.hstack((Anomali, ii))

        if Anomali.shape[0] != 0:
            for ii in range(Anomali.shape[0]):
                count_cluster = count_cluster + 1
                indiceAnomali = np.hstack((indiceAnomali, np.where(NKM.T == Anomali[ii])[1]))
                # label_total[label_DATA[np.where(NKM.T == Anomali[ii])[1]]] = count_cluster
                label_total[label_DATA[np.where(NKM.T == Anomali[ii])[1]]] = 1

            C_total = np.append(C_total, NC[Anomali][:], axis=0)

        # controllare cosa fare
        aux_index = np.delete(aux_index, Anomali)
        count_cluster = count_cluster + 1
        for it in aux_index:
            card[it] = np.sum(NKM.T == it)
            NC_mean = np.sum([NC_mean, NC[it][:] * card[it]], axis=0)
        if aux_index.shape[0] != 0:
            NC_mean = NC_mean / np.sum(card)
            C_total = np.append(C_total, NC_mean, axis=0)
            # label_total[label_total == -1] = count_cluster
            label_total[label_total == -1] = 0

        u_old, d = _init_memberships(X, C_total, distance)
        um = u_old ** m
        eta_total = _eta_update(um, d)
    else:
        C_total = centers
        label_total = labels
        eta_total = eta

    return C_total, label_total, eta_total
'''
def check_anomali_pcm(X, centers, labels, sigma, distance,m):

    NKM, NC = _no_empty_clusters(centers, labels)

    u_old, d = _init_memberships(X, NC, distance)
    um = u_old ** m
    eta_new = _eta_update(um, u_old, d)




    n_cluster_no_empty = int(np.max(NKM))+1
    card_NKM = np.empty((n_cluster_no_empty))
    for it in np.arange(n_cluster_no_empty):
        card_NKM[it] = (NKM==it).sum()



    DC = cdist(NC, NC, distance)
    DC_new = DC.copy()
    for it in np.arange(DC_new.shape[0]):
        DC_new[it,it]=100*sigma
    NKM_new = NKM.copy()
    NC_new = NC.copy()
    card_NKM_new = card_NKM.copy()

    if max(NKM) > 1:


        aux_DC = DC_new< sigma

        count_cluster=0
        elenco_cluster = np.arange(n_cluster_no_empty)


        while aux_DC.size-aux_DC.sum()<aux_DC.size:
            indici_min = np.where(DC_new == np.min(DC_new))[0]
            to_keep = indici_min[0]
            to_drop = indici_min[1]
            NKM_new[NKM_new==to_drop] = to_keep
            elenco_cluster=np.delete(elenco_cluster,to_drop)
            NC_new[to_keep,:] = \
                (card_NKM_new[to_keep]*NC[to_keep,:]+card_NKM_new[to_drop]*NC[to_drop,:] )/ (card_NKM_new[to_keep] + card_NKM_new[to_drop])
            card_NKM_new[to_keep] = (card_NKM_new[to_keep] + card_NKM_new[to_drop])
            NC_new = np.delete(NC_new, to_drop, 0)
            NKM_new, NC_new = _no_empty_clusters(NC_new, NKM_new)
            DC_new = cdist(NC_new, NC_new, distance)
            u_old, d = _init_memberships(X, NC_new, distance)
            um = u_old ** m
            eta_new = _eta_update(um, u_old, d)
            for it in np.arange(DC_new.shape[0]):
                DC_new[it, it] = 100 * sigma
            aux_DC = DC_new < sigma

    return  NC_new, NKM_new,eta_new


class FKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters, distance, constraint = 'probabilistic', m = 2.0, eta=None, sigma=0.5,
                 init='k-means++', n_init=10, max_iter=100,
                 tol_memberships=1e-3, tol_centroids=1e-4):

        self.n_clusters = n_clusters
        self.m = m
        self.init = init
        self.max_iter = max_iter
        self.tol_memberships = tol_memberships
        self.tol_centroids = tol_centroids
        self.n_init = n_init
        self.constraint = constraint
        self.distance = distance
        self.sigma = sigma
        self.eta = eta

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] == 1:
            X = X.T

        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr')
        if X.shape[0] == 1:
            X = X.T
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        if X.dtype.kind != 'f':
            warnings.warn("Got data type %s, converted to float "
                          "to avoid overflows" % X.dtype,
                          RuntimeWarning, stacklevel=2)
            X = X.astype(np.float)

        return X

    def fit(self, X, Y=None):
        """Compute fuzzy c-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """



        if Y is None:
            self.centers = _init_centroids(X,
                                       self.n_clusters,
                                       init=self.init,
                                       random_state=None,
                                       x_squared_norms=row_norms(X, squared=True))
        else:
            n_labels = int(np.max(Y))
            self.centers = np.zeros([n_labels + 1, np.shape(X)[1]])
            for l in np.arange(n_labels + 1):
                self.centers[l, :] = np.mean(X[np.where(Y==l)[0],:],axis=0)

        '''
        self.centers = _init_centroids(X,
                                       self.n_clusters,
                                       init=self.init,
                                       random_state=None,
                                       x_squared_norms=row_norms(X, squared=True))
        '''
        u, d = _init_memberships(X, self.centers, self.distance)

        cluster_centers, predicted_labels = \
            f_k_means(X,
                      n_clusters=self.n_clusters,
                      m=self.m,
                      tol_memberships=self.tol_memberships,
                      tol_centroids=self.tol_centroids,
                      max_iter=self.max_iter,
                      init=self.centers,
                      constraint=self.constraint,
                      distance=self.distance,
                      n_init=self.n_init,
                      eta=self.eta,
                      sigma=self.sigma)
        self.labels_ = predicted_labels
        self.cluster_centers_ = cluster_centers

        return self

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




class PKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, distance = 'euclidean', n_clusters = 4, constraint = 'possibilistic', tol_centroids=1e-4,
                 max_iter=100, m=2.0, init='k-means++',
                 tol_memberships=1e-8, n_init=10, sigma=0.5, eta=None):

        self.n_clusters = n_clusters
        self.m = m
        self.eta = eta
        self.init = init
        self.max_iter = max_iter
        self.tol_memberships = tol_memberships
        self.tol_centroids = tol_centroids
        self.n_init = n_init
        self.constraint = constraint
        self.distance = distance
        self.sigma = sigma

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] == 1:
            X = X.T

        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr')
        if X.shape[0] == 1:
            X = X.T
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        if X.dtype.kind != 'f':
            warnings.warn("Got data type %s, converted to float "
                          "to avoid overflows" % X.dtype,
                          RuntimeWarning, stacklevel=2)
            X = X.astype(np.float)

        return X


    def fit(self, X, Y=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """



        if Y is None:
            self.centers = _init_centroids(X,
                                       self.n_clusters,
                                       init=self.init,
                                       random_state=None,
                                       x_squared_norms=row_norms(X, squared=True))
        else:
            n_labels = int(np.max(Y))
            self.centers = np.zeros([n_labels + 1, np.shape(X)[1]])
            for l in np.arange(n_labels + 1):
                self.centers[l, :] = np.mean(X[np.where(Y==l)[0],:],axis=0)


        u, d = _init_memberships(X, self.centers, self.distance)


        cluster_centers, predicted_labels, eta = \
            p_k_means(X,
                      n_clusters=self.n_clusters,
                      m=self.m,
                      u = u, centers = self.centers,d=d,
                      tol_memberships=self.tol_memberships,
                      tol_centroids=self.tol_centroids,
                      max_iter=self.max_iter,
                      distance=self.distance,
                      sigma=self.sigma,
                      eta = self.eta)




        self.labels_ = predicted_labels
        self.cluster_centers_ = cluster_centers
        self.eta = eta





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
        u = _memberships_update_possibilistic(d, self.m, self.eta)
        predicted_labels = _labels_computation(u)

        return predicted_labels