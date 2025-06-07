import numpy as np
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from coclust.clustering.spherical_kmeans import SphericalKmeans
import random
from collections import Counter
from .ckmeans import KMeansChi2

class Rcwk:
    def __init__(self, k, lambda_val=0.0001, alpha=0, beta=4, tmax=30, chi2=True, n_init=20, scale=False, init='r', z=None, 
                 scaling_factor=0.5, lambda_val_2=1):
        self.k = k
        self.lambda_val = lambda_val
        self.alpha = alpha 
        self.beta = beta
        self.tmax = tmax
        self.chi2 = chi2
        self.n_init = n_init
        self.scale = scale
        self.init = init
        self.z = z
        self.scaling_factor = scaling_factor
        self.P2_mat = []
        self.lambda_val_2 = lambda_val_2

    def fit(self, X, verbose=False):
        n, p = X.shape
        scaling_factor = self.scaling_factor
        colsum = X.sum(axis=0) if self.chi2 else None
        X_rowsums = X.sum(axis=1) if self.chi2 else None

        if self.z is None:
            if self.init == 's':
                sk = SphericalKmeans(n_clusters=self.k, n_init=self.n_init)
                sk.fit(X)
                labels = np.array(sk.row_labels_)
            elif self.init == 'r':
                labels = np.random.randint(0, self.k, size=n)
            elif self.init == 'k':
                labels = KMeans(n_clusters=self.k, n_init=self.n_init).fit(X).labels_
            elif self.init == 'chi2':
                kmc = KMeansChi2(k=self.k)
                kmc.fit(X)
                labels = kmc.labels_
            labels = labels.reshape(-1, 1)
        else:
            labels = self.z.reshape(-1, 1)

        dist = np.zeros(self.k)
        D = np.zeros(p)
        on = np.ones(p)
        lambda_val = self.lambda_val / p**2

        if self.scale:
            X = preprocessing.scale(X, with_mean=False)

        variances = self._compute_var_disp(X, labels, self.k)
        X = X / variances
        M = np.zeros((self.k, p))
        for i in range(self.k):
            I = np.where(labels == i)[0]
            M[i, :] = X[I, :].sum(axis=0) if self.chi2 else np.mean(X[I, :], axis=0)

        for i in range(self.k):
            I = np.where(labels == i)[0]
            for j in I:
                if not self.chi2:
                    D += self._vec_euc_dist(X[j, :], M[i, :], on)
                else:
                    D += self._vec_custom_distance_cached(X[j, :], M[i, :], colsum, X_rowsums[j], M[i, :].sum(), on)

        beta = self.beta
        if self.alpha != 0:
            alpha = self.alpha
        else:
            denom_alpha = np.sum((1 / (beta * D)) ** (1.0 / (beta - 1)))
            alpha = (1.0 / denom_alpha) ** (beta - 1)

        weight = np.zeros(p)
        w_dummy = np.zeros((p, self.k))
        for l in range(p):
            denom = np.sum((D[l] / D) ** (1.0 / (beta - 1)))
            weight[l] = 1.0 / denom

        for iter in range(self.tmax):
            for i in range(self.k):
                I = np.where(labels == i)[0]
                if len(I) == 0:
                    farthest_point = np.argmax(distances)
                    labels[farthest_point] = i
                    I = [farthest_point]
                    distances[farthest_point] = -1
                M[i, :] = X[I, :].sum(axis=0) if self.chi2 else np.mean(X[I, :], axis=0)

            D = np.zeros(p)
            for i in range(self.k):
                I = np.where(labels == i)[0]
                for j in I:
                    if not self.chi2:
                        D += self._vec_euc_dist(X[j, :], M[i, :], on)
                    else:
                        D += self._vec_custom_distance_cached(X[j, :], M[i, :], colsum, X_rowsums[j], M[i, :].sum(), on)
                
            for i in range(p):
                x_thresh = alpha / D[i]
                x_thresh = max(0, x_thresh - lambda_val)
                weight[i] = (1 / self.beta * x_thresh) ** (1.0 / (self.beta - 1)) if x_thresh > 0 else 0

            w_dummy = weight**self.beta + lambda_val * weight

            P2 = 0
            distances = []
            for i in range(n):
                for j in range(self.k):
                    if not self.chi2:
                        dist[j] = self._wt_euc_dist(X[i, :], M[j, :], w_dummy)
                    else:
                        dist[j] = self._custom_distance_cached(X[i, :], M[j, :], colsum, X_rowsums[i], M[j, :].sum(), w_dummy)
                labels[i] = np.argmin(dist)
                distances.append(dist[labels[i]])
                P2 += dist[labels[i][0]]

            P2 = P2 / n - alpha * np.sum(weight)
            self.P2_mat.append(P2)
            if verbose:
                print(P2)
            if iter == 0:
                P1 = P2
            elif P1 == P2:
                break
            else:
                P1 = P2

        L = np.sum(weight > 0)

        self.labels_ = [i[0] for i in labels]
        self.weights_ = weight
        self.L = L
        self.alpha = alpha
        self.P2 = P2

    def _compute_var_disp(self, X, labels, k):
        n, p = X.shape
        var_disp = np.zeros(p)
        cluster_vars = np.zeros((k, p))
    
        for i in range(k):
            cluster_points = X[labels.ravel() == i]
            if cluster_points.shape[0] > 1:
                cluster_vars[i] = np.var(cluster_points, axis=0)
            else:
                cluster_vars[i] = 0.0  # Avoid NaNs
    
        var_disp = np.var(cluster_vars, axis=0)
        return var_disp


    def _custom_distance_cached(self, xi, mj, colsum, x_rowsum, m_rowsum, w):
        term1 = xi / (np.sqrt(colsum)*x_rowsum) 
        term2 = (mj/m_rowsum) / np.sqrt(colsum) 
        
        return np.sum((term1 - term2) ** 2 * w)

    def _vec_custom_distance_cached(self, xi, mj, colsum, x_rowsum, m_rowsum, w):
        term1 = xi / (np.sqrt(colsum)*x_rowsum) 
        term2 = (mj/m_rowsum) / np.sqrt(colsum) 
        
        return (term1 - term2) ** 2 * w

    def get_top_features(self, feature_names, n_features=10):
        weigh_word = np.sum(self.weights_, axis=1)
        weigh_word = sorted(zip(feature_names, weigh_word), key=lambda x: x[1], reverse=True)
        return [i[0] for i in weigh_word[:n_features]]

    def _vec_euc_dist(self, x1, x2, w):
        return w * ((x1 - x2) ** 2)

    def _wt_euc_dist(self, x1, x2, w):
        return np.sum(w * ((x1 - x2) ** 2))