# -*- coding: utf-8 -*-

"""
coclust_dcc.py
"""

# Author: Severine Affeldt <severine.affeldt@gmail.com>
#         Lazhar Labiod <l.labiod@gmail.com>
#         Mohamed Nadif <mohamed.nadif@u-paris.fr>            

# License: BSD 3 clause
import sys
import itertools
import numpy as np
import random
import scipy.sparse as sp
from joblib import Parallel, delayed
import multiprocessing

from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import Counter

from sklearn.utils import check_random_state, check_array
from sklearn.metrics import confusion_matrix
from .base_diagonal_coclust import BaseDiagonalCoclust
from sklearn.cluster import KMeans

import copy

from numpy.linalg import matrix_power

def dcc_sample_function(prob):
    return random.choices(np.arange(len(prob)), k = 1, weights = prob)

class CoclustDcc(BaseDiagonalCoclust):
    """Directional Co-Clustering based on the block von Mises-Fisher mixture model.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    init_row : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial row labels

    init_col : numpy array or scipy sparse matrix, \
        shape (n_var, n_clusters), optional, default: None
        Initial column labels

    max_iter : int, optional, default: 100
        Maximum number of iterations

    n_stoch : int, optional, default: 70
        Maximum number of stochastic iterations

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs in terms of modularity.

    tol : float, default: 1e-9
        Relative tolerance with regards to modularity to declare convergence

    Attributes
    ----------
    n_iter_ : int
        Total number of iterations

    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row

    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column

    best_criterion_value : float
        Final value of the criterion for the best partionning among all initializations

    all_criterion_values : list
        All criterion values for the best partionning among all initializations

    References
    ----------
    * Salah, A. and Nadif, M., Directional co-clustering. \
    Advances in Data Analysis and Classification 2019: , 13(3): 591–620.
    """

    def __init__(self, n_clusters = 2, mat_supp = None, init_row = None, init_col = None, max_iter = 100, 
                 n_stoch = 70, n_init = 20, true_labels = None, computeMod = False, tol=1e-9):
        self.n_clusters = n_clusters
        self.init_row = init_row
        self.init_col = init_col
        self.max_iter = max_iter
        self.n_stoch = n_stoch
        self.n_init = n_init
        self.tol = tol
        self.true_labels = true_labels

        self.n_iter_ = None
        self.row_labels_ = None
        self.column_labels_ = None
        self.best_criterion_value = np.inf
        self.all_criterion_values = []
        self.best_init = None
        self.total_supp = None
        self.trace_crit = None
        self.mod = -np.inf
        self.computeMod = computeMod
        self.Z_part = None
        self.W_part = None
        
        self.all_RRun = []
        self.num_cores = multiprocessing.cpu_count()
        
        self.mat_supp = mat_supp
        
    def _fit_single_with_init(self, X, i = -1, verbose = False):
        
        # row partition preparation
        row_c = None
        if self.init_row is None:
            row_c = np.random.randint(self.n_clusters, size = X.shape[0])
        else:
            if isinstance(self.init_row, np.ndarray):
                row_c = [int(self.init_row[i, idx]) for idx in range(self.init_row.shape[1])]
            else:
                sys.exit("-- ERR --> row_init is not a matrix")
        # col partition preparation
        col_c = None
        if self.init_col is None:
            col_c = np.random.randint(self.n_clusters, size = X.shape[1])
        else:
            if isinstance(self.init_col, np.ndarray):
                col_c = [int(self.init_col[i, idx]) for idx in range(self.init_col.shape[1])]
            else:
                sys.exit("-- ERR --> col_init is not a matrix")
    
        ###
        if verbose == True:
            print("# --/-- DCC init", i, "--/--")
        RRun = self._fit_single(X = X, sgl_row_c = row_c, sgl_col_c = col_c, verbose = verbose)
        return RRun

    def _fit_single(self, X, sgl_row_c, sgl_col_c, verbose = False):
        """Perform one run of directional co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        vtw = list()

        # Compute initial binary column-cluster indicator matrix
        W = np.zeros((X.shape[1], self.n_clusters))
        W = sp.lil_matrix(W)
        W[np.arange(X.shape[1]), sgl_col_c] = 1
        W = sp.csr_matrix(W)
        
        # Compute initial row centroids 
        my_counter = Counter(sgl_col_c)
        my_counter = [1.0/np.sqrt(my_counter[j]) for j in range(len(my_counter))]
        MU_w = np.diag(my_counter)
        MU_w = sp.csr_matrix(MU_w)
        
        # Column centroids MU^z
        my_counter = Counter(sgl_row_c)
        my_counter = [1.0/np.sqrt(my_counter[j]) for j in range(len(my_counter))]
        MU_z = np.diag(my_counter)
        MU_z = sp.csr_matrix(MU_z)
        
        prev_W = None
        Prev_Z = None
        prev_row_partition = None
        prev_col_partition = None
        vanishing_clust = False
        dcc_modularity = -np.inf

        #if self.computeMod == True:
        #    # (1) Compute the indep modularity matrix
        #    print("Compute the indep modularity matrix")
        #    row_sums = np.matrix(X.sum(axis=1))
        #    col_sums = np.matrix(X.sum(axis=0))
        #    N = float(X.sum())
        #    indep = (row_sums.dot(col_sums)) / N
        #    B = (1/N)*(X - indep)
        #    B = sp.csr_matrix(B)
            

        ### DCC alternating optimization
        
        for curr_iter in range(self.max_iter):

            # row partitionning
            Zt = ( self.mat_supp[0] @ W ) @ ( MU_w @ MU_z )
            Zt = normalize(Zt, axis = 0, copy = False, norm = "l2")
            Zt = sp.csr_matrix(Zt)
            
            row_partition = list(itertools.chain.from_iterable(Zt.argmax(axis = 1).tolist()))
            Z = np.zeros((X.shape[0], self.n_clusters))
            Z = sp.lil_matrix(Z)
            Z[np.arange(len(row_partition)), row_partition] = 1
            Z = sp.csr_matrix(Z)
            
            # Update column centroids MU^z
            my_counter = Counter(row_partition)
            if len(my_counter) != self.n_clusters:
                vanishing_clust = True
                print("Vanishing row cluster")
                print(my_counter)
                break
            
            my_counter = [1.0/(np.sqrt(my_counter[j])+np.finfo(float).eps) for j in range(len(my_counter))]
            MU_z = np.diag(my_counter)
            MU_z = sp.csr_matrix(MU_z)
            
            # Column partitionning
            Wt = (self.mat_supp[1].T @ (Z @ (MU_z @ MU_w)))            
            Wt = normalize(Wt, axis = 0, copy = False, norm = "l2")
            Wt = sp.csr_matrix(Wt)
            
            col_partition = None
            if curr_iter <= self.n_stoch:
                col_partition =  np.apply_along_axis(func1d = dcc_sample_function, axis = 1, arr = Wt.toarray())
            else:
                col_partition = Wt.argmax(axis = 1).tolist()
            
            col_partition = list(itertools.chain.from_iterable(col_partition))
            W = np.zeros((X.shape[1], self.n_clusters))
            W = sp.lil_matrix(W)
            W[np.arange(len(col_partition)), col_partition] = 1
            W = sp.csr_matrix(W)

            # Update row centroids MU^w
            my_counter = Counter(col_partition)
            if len(my_counter)  != self.n_clusters:
                vanishing_clust = True
                print("Vanishing col cluster")
                print(my_counter)
                break
            my_counter = [1.0/(np.sqrt(my_counter[j])+np.finfo(float).eps) for j in range(len(my_counter))]
            MU_w = np.diag(my_counter)
            MU_w = sp.csr_matrix(MU_w)
            
            # Compute the DCC citeria (pos)
            vtw.append(np.sum(Z.multiply( (( self.mat_supp[0] @ W ) @ ( MU_w @ MU_z )) ) ))
            
            # --!!-- KEEP for DEV --!!--
            #if self.true_labels is not None and verbose == True:
            #    curr_nmi = normalized_mutual_info_score(self.true_labels, row_partition, average_method="arithmetic")
            #    curr_ari = adjusted_rand_score(self.true_labels, row_partition)
            #    
            #    ##l2 normalization
            #    W = normalize(W, axis = 0, copy = False, norm = "l2")
            #    Z = normalize(Z, axis = 0, copy = False, norm = "l2")
            #    tmp_crit = ((0.5*(Z @ ((W.T @ W) @ Z.T))).diagonal().sum() - ((self.mat_supp @ W) @ Z.T).diagonal().sum())
            #
            #    print("# --> iter {:.0f}\t [curr_NMI {:.4f}] [curr_ARI {:.4f}] [curr_vtw {:.4f}] [n_vtw {:.4f}]".format(curr_iter, curr_nmi, curr_ari, vtw[curr_iter], tmp_crit))
            # --!!-- KEEP for DEV --!!--
            
            if curr_iter > 0:
                if abs(vtw[curr_iter]-vtw[curr_iter-1]) < self.tol:
                    break
            
        ref_idx_iter = None
        ref_idx_iter_ll = None
        if len(vtw)>0:
            ref_idx_iter = (len(vtw)-1)
            print("------------>>!! END DCC: ref_idx_iter =", ref_idx_iter)
            ref_idx_iter_ll = vtw[ref_idx_iter]

        #if self.computeMod == True:
        #    # (2) Compute modularity
        #    print("Compute the modularity value")
        #    BW = B.dot(W)
        #    #BtZ = (B.T).dot(Z)
        #    k_times_k = (Z.T).dot(BW)
        #    dcc_modularity = np.trace(k_times_k.toarray())
                
        # Evaluate the row partition
        curr_ari = -1
        curr_nmi = -1
        curr_acc = -1
        if self.true_labels is not None:
            curr_ari = adjusted_rand_score(self.true_labels, row_partition)
            curr_nmi = normalized_mutual_info_score(self.true_labels, row_partition, average_method="arithmetic")            
        
        # Compute the trace criteria and append it to the DCC criteria list (neg)
        #print("------------>>!! compute crit")
        W_noNorm = W
        Z_noNorm = Z
        
        W = normalize(W, axis = 0, copy = False, norm = "l2")
        Z = normalize(Z, axis = 0, copy = False, norm = "l2")
        #tmp_crit = 0.5*((X @ X.T).diagonal().sum() - 2*((X @ W) @ Z.T).diagonal().sum() + (Z @ ((W.T @ W) @ Z.T)).diagonal().sum())
        #vtw[ref_idx_iter] = ((0.5*(Z @ ((W.T @ W) @ Z.T))).diagonal().sum() - ((self.mat_supp @ W) @ Z.T).diagonal().sum())
        #vtw[ref_idx_iter] = ((0.5*(Z @ ((W.T @ W) @ Z.T))).diagonal().sum() - ((self.mat_supp[0] @ W) @ Z.T).diagonal().sum())
        vtw.append((0.5*(Z @ ((W.T @ W) @ Z.T))).diagonal().sum() - ((self.mat_supp[0] @ W) @ Z.T).diagonal().sum())

        #print("# >>> s_iter {:.0f}\t [s_NMI {:.4f}] [s_ARI {:.4f}] [silh_r {:.4f}] [n_ll {:.4f}] [ll {:.4f}], {:.0f} {:.0f}".format(-1, curr_nmi, curr_ari, curr_silh_r, vtw[ref_idx_iter], ref_idx_iter_ll, ref_idx_iter, ref_idx_iter))
        if ref_idx_iter is not None:
            #print("# >>> s_iter {:.0f}\t [s_NMI {:.4f}] [s_ARI {:.4f}] [crit {:.4f}] [ll {:.4f}], [m {:.4f}], {:.0f}".format(-1, curr_nmi, curr_ari, vtw[ref_idx_iter+1], ref_idx_iter_ll, dcc_modularity, ref_idx_iter))
            print("# >>> s_iter {:.0f}\t [s_NMI {:.4f}] [s_ARI {:.4f}] [crit {:.4f}] [ll {:.4f}], {:.0f}".format(-1, curr_nmi, curr_ari, vtw[ref_idx_iter+1], ref_idx_iter_ll, ref_idx_iter))
        
        if vanishing_clust == False:
            return {"rowcluster": row_partition, "colcluster": col_partition, "ll": vtw,
                    "iter": ref_idx_iter, "dcc_crit":ref_idx_iter_ll, "dcc_mod":dcc_modularity,
                    "Z_part": Z_noNorm, "W_part": W_noNorm}
        else:
            return None

    def fit(self, X, verbose = False):
        """Perform directional co-clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
      
        check_array(X, accept_sparse = True, dtype = "numeric", order = None,
                    copy = False, force_all_finite = True, ensure_2d = True,
                    allow_nd = False, ensure_min_samples = self.n_clusters,
                    ensure_min_features = self.n_clusters,
                    #warn_on_dtype = False, 
                    estimator = None)
      
        # Coherence tests
        # ----
        if self.n_stoch >= self.max_iter:
            sys.exit("# -- ERR --> Error stoch_iter.max must be less than iter.max")
      
        if self.n_clusters >= np.min(X.shape):
            sys.exit("# -- ERR --> More clusters than distinct objects/features (rows/columns)")
        
        #if self.n_init > 0:
        if self.init_row is not None:
            # Coherence tests for init_row
            if isinstance(self.init_row, np.ndarray):
                if self.init_row.shape[0] < self.n_init:
                    sys.exit("# -- ERR --> Less row partitions than n_init")
                if self.init_row.shape[1] != X.shape[0]:
                    sys.exit("# -- ERR --> The length of the row partitions is different from the number of objects")
            else:
                sys.exit("# -- ERR --> The row partitions should be given in a matrix format")
        if self.init_col is not None:
            # Coherence tests for init_col
            if isinstance(self.init_col, np.ndarray):
                if self.init_col.shape[0] < self.n_init:
                    sys.exit("# -- ERR --> Less column partitions than n_init")
                if self.init_col.shape[1] != X.shape[1]:
                    sys.exit("# -- ERR --> The length of the col partitions is different from the number of columns")
            else:
                sys.exit("# -- ERR --> The column partitions should be given in a matrix format")
       
        # normalize rows to have unit L2 norm
        # ----
        
        X = sp.csr_matrix(X)
       
        X =  normalize(X, norm = "l2", axis = 1)
      
        inputs = range(self.n_init)
        
        results = Parallel(n_jobs = self.num_cores)(delayed(self._fit_single_with_init)(X, i, verbose) for i in inputs)
       
        self.all_RRun.extend(results)
        self.all_RRun = list(filter(None, self.all_RRun)) 

        self.best_criterion_value = -np.inf
        #for i in range(self.n_init):
        for i in range(len(self.all_RRun)):
            curr_Run = self.all_RRun[i]
                            
            if curr_Run["dcc_crit"] > self.best_criterion_value:
                print(curr_Run["dcc_crit"], self.best_criterion_value)
                self.best_criterion_value = curr_Run["dcc_crit"]
                self.best_init = i

        if self.best_init is not None:
            Run = self.all_RRun[self.best_init]
            # Update all attributes
            self.n_iter_ = Run["iter"]
            self.row_labels_ = Run["rowcluster"]
            self.column_labels_ = Run["colcluster"]
            self.best_criterion_value = Run["dcc_crit"]
            self.all_criterion_values = Run["ll"]
            self.trace_crit = Run["ll"][self.n_iter_+1]
            #self.mod = Run["dcc_mod"]

            self.Z_part = Run["Z_part"]
            self.W_part = Run["W_part"]

            if self.computeMod == True:
                # (1) Compute the indep modularity matrix
                #print("Compute the indep modularity matrix")
                row_sums = np.matrix(X.sum(axis=1))
                col_sums = np.matrix(X.sum(axis=0))
                N = float(X.sum())
                indep = (row_sums.dot(col_sums)) / N
                B = (1/N)*(X - indep)
                B = sp.csr_matrix(B)

                # (2) Compute modularity
                #print("Compute the modularity value")
                BW = B.dot(self.W_part)
                #BtZ = (B.T).dot(Z)
                k_times_k = (self.Z_part.T).dot(BW)
                self.mod = np.trace(k_times_k.toarray())
        
        return self

    def get_assignment_matrix(self, kind, i):
        """Returns the indices of 'best' i cols of an assignment matrix
        (row or column).

        Parameters
        ----------
        kind : string
             Assignment matrix to be used: rows or cols

        Returns
        -------
        numpy array or scipy sparse matrix
            Matrix containing the i 'best' columns of a row or column
            assignment matrix
        """
        if kind == "rows":
            s_bw = np.argsort(self.bw)
            return s_bw[:, -1:-(i+1):-1]
        if kind == "cols":
            s_btz = np.argsort(self.btz)
            return s_btz[:, -1:-(i+1):-1]
