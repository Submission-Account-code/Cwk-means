import numpy as np
import pandas as pd


def _top_terms(words, weights, N):
    sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)  
    top_words = [words[i] for i in sorted_indices[:N]] 
    top_weights = [weights[i] for i in sorted_indices[:N]] 
    
    return dict(zip(top_words, top_weights))


def _extract_cluster_matrices(doc_term_matrix, labels):
    labels = np.array(labels)
    cluster_matrices = {}
    for cluster_label in range(len(set(labels))):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_matrix = doc_term_matrix[cluster_indices, :]
        cluster_matrices[cluster_label] = cluster_matrix
    return cluster_matrices

def get_top_features(vocabulary, weights, n_features=20):
    # Sort terms with weights and convert to dictionary
    weigh_word_dict = {term: weight for term, weight in sorted(zip(vocabulary, weights), key=lambda x: x[1], reverse=True)}
    # Select the top n_features from the sorted dictionary
    top_features = {term: weigh_word_dict[term] for term in list(weigh_word_dict.keys())[:n_features]}
    return top_features

def get_top_terms_count(mat, labels, vocabulary, n_features=10):
    counts = _extract_cluster_matrices(mat, labels)
    result = {}  
    for i in range(len(set(labels))):
        count = counts[i].sum(axis=0).tolist()
        weigh_word = sorted(zip(vocabulary, count), key=lambda x: x[1], reverse=True)
        cluster_terms = {} 
        for term in weigh_word[:n_features]:
            cluster_terms[term[0]] = round(term[1], 2)
        result[f'Cluster {i}'] = cluster_terms 
    return result

def get_top_terms_weight(mat, labels, weights, vocabulary, max_count=500, n_features=10):
    df = pd.DataFrame(mat)
    df['Cluster'] = labels
    # Top terms by count
    word_freq_by_cluster = df.groupby('Cluster').sum()
    top_terms_count = np.argsort(-word_freq_by_cluster.values, axis=1)
    # Top terms by weight
    top_terms = {}
    for cluster, top_words in enumerate(top_terms_count):
        #term_weights = {vocabulary[word]: round(weights[word], 6) for word in top_words[:max_count]}
        term_weights = {vocabulary[word]: weights[word] for word in top_words[:max_count]}
        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:n_features]
        top_terms[f'Cluster {cluster}'] = dict(sorted_terms)
    
    return top_terms
