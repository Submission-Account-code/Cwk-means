import numpy as np
import random
from sklearn.cluster import KMeans
from coclust.clustering.spherical_kmeans import SphericalKmeans

class KMeansChi2:
    def __init__(self, k, tmax=30, n_init=20, init='s', tol=1e-6):
        self.k = k
        self.tmax = tmax
        self.n_init = n_init
        self.init = init
        self.tol = tol

    def fit(self, X, verbose=False):
        n, p = X.shape
        self.inertia_tab = []
        colsum = X.sum(axis=0)  # Pré-calculé une seule fois

        if self.init == 'r':
            labels = np.random.randint(0, self.k, size=n)
        elif self.init == 'k':
            labels = KMeans(n_clusters=self.k, n_init=self.n_init).fit(X).labels_
        elif self.init == 's':
            sk = SphericalKmeans(n_clusters=self.k, n_init=self.n_init)
            sk.fit(X)
            labels = np.array(sk.row_labels_)
        else:
            raise ValueError("Init method not recognized")

        labels = labels.reshape(-1, 1)
        previous_inertia = np.inf

        for iter in range(self.tmax):
            # Calcul des centres
            M = np.zeros((self.k, p))
            m_rowsums = np.zeros(self.k)
            for i in range(self.k):
                I = np.where(labels.flatten() == i)[0]
                if len(I) == 0:
                    continue
                M[i, :] = X[I, :].sum(axis=0)
                m_rowsums[i] = M[i, :].sum()

            # Mise à jour des labels
            new_labels = np.zeros(n, dtype=int)
            for i in range(n):
                xi = X[i, :]
                x_rowsum = xi.sum()
                dist = np.zeros(self.k)
                for j in range(self.k):
                    if m_rowsums[j] == 0:
                        dist[j] = np.inf
                    else:
                        dist[j] = self._custom_distance(
                            xi, x_rowsum, M[j, :], m_rowsums[j], colsum, verbose=verbose
                        )
                new_labels[i] = np.argmin(dist)

            labels = new_labels.reshape(-1, 1)

            # Calcul de l'inertie
            inertia = 0.0
            for i in range(n):
                cluster_idx = labels[i, 0]
                if m_rowsums[cluster_idx] == 0:
                    continue
                xi = X[i, :]
                x_rowsum = xi.sum()
                inertia += self._custom_distance(
                    xi, x_rowsum, M[cluster_idx], m_rowsums[cluster_idx], colsum, verbose=verbose
                )
            self.inertia_tab.append(inertia)

            if verbose:
                print('*'*30)
                print(f"Itération {iter+1}, Inertie : {inertia:.6f}")

            if abs(previous_inertia - inertia) < self.tol:
                if verbose:
                    print("Arrêt : convergence de l'inertie atteinte.")
                break

            previous_inertia = inertia

        self.labels_ = labels.flatten()
        self.inertia_ = inertia

    def _custom_distance(self, xi, x_rowsum, mj, m_rowsum, colsum, verbose=False):
        sqrt_colsum = np.sqrt(colsum)
        term1 = xi / (sqrt_colsum * x_rowsum)
        term2 = (mj / m_rowsum) / sqrt_colsum

        if verbose:
            print(f'term 1: {term1}')
            print(f'term 2: {term2}')

        dist = np.sum((term1 - term2) ** 2)
        return dist
