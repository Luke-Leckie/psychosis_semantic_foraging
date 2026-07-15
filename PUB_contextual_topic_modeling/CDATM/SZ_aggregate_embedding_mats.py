#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
from gensim.models import coherencemodel
import pickle
import os
from gensim import corpora, models, similarities 
from gensim.models import Word2Vec, KeyedVectors
from ksvd import ApproximateKSVD 
import random
import signal
import numpy as np
random.seed(42)
import sys
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from itertools import combinations
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import networkx as nx
nltk.download('stopwords')


def remove_unwanted_tokens(embedding_matrix, all_tokens):
    """
    Remove CLS, SEP, stopwords, tokens containing '#' and tokens made entirely of non-alphanumeric characters.

    Args:
    - embedding_matrix: A NumPy array of shape (num_tokens, embedding_dim).
    - all_tokens: A NumPy array of tokens of shape (num_tokens,).

    Returns:
    - filtered_embedding_matrix: A NumPy array of the remaining embeddings.
    - filtered_tokens: A NumPy array of the remaining tokens.
    """
    
    stop_words = set(stopwords.words('english'))
    unwanted_tokens = {'[CLS]', '[SEP]', '[PAD]', '<pad>','<s>', '</s>','âĢ¦','Ġ','ĠâĢ¦', None}

    def is_valid_token(token):
        if token is None:
            return False
        if token in unwanted_tokens:
            return False
        if token.lower() in stop_words:
            return False
        if '#' in token:
            return False
        if not any(char.isalnum() for char in token):
            return False
        return True

    mask = np.array([is_valid_token(token) for token in all_tokens])
    filtered_embedding_matrix = embedding_matrix[mask]
    filtered_tokens = all_tokens[mask]
    print(list(set(filtered_tokens)))

    return filtered_embedding_matrix, filtered_tokens




def get_word_graph(token, corpus_embedding_matrix, corpus_token_matrix,
                   power=2.0, k=4):
    # 1) find all positions of `token`
    matched = [i for i, t in enumerate(corpus_token_matrix) if t == token]
    n = len(matched)
    G = nx.Graph()
    G.add_nodes_from(matched)
    if n <= 1:
        return G

    # 2) extract just those embeddings
    X = corpus_embedding_matrix[matched]

    # 3) fit a cosine‐NN on X
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", n_jobs=-1).fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    # 5) build the graph from the top‐k neighbors
    for i, (row_d, row_i) in enumerate(zip(dists, idxs)):
        for dist, j in zip(row_d[1:], row_i[1:]):  # skip the self‐match at [0]
            sim = 1 - dist                         # back to cosine‐similarity
            if power != 1.0:
                sim = sim ** power
            if sim > 0:
                G.add_edge(matched[i], matched[j], weight=sim)
    return G


def get_components(token, corpus_embeddings, corpus_tokens,
                   threshold=0.5, power=1.0):
    # 1) Find all positions of the token
    matched_idx = [i for i, t in enumerate(corpus_tokens) if t == token]
    if not matched_idx:
        return [], 0
    if len(matched_idx) == 1:
        return [corpus_embeddings[matched_idx[0]]], 1
    n=len(matched_idx)
    # 2) Subset embeddings
    X = corpus_embeddings[matched_idx]

    # 3) Build NN model: if no k given, connect to all others
    n_neighbors = 5
    nn = NearestNeighbors(n_neighbors=min(n, n_neighbors),
                          metric="cosine",
                          n_jobs=-1).fit(X)

    # 4) Query neighbors
    dists, idxs = nn.kneighbors(X, return_distance=True)

    # 5) Build thresholded graph
    G = nx.Graph()
    G.add_nodes_from(matched_idx)
    for i, (d_row, i_row) in enumerate(zip(dists, idxs)):
        for dist, j_local in zip(d_row[1:], i_row[1:]):
            sim = 1 - dist
            if power != 1.0:
                sim = sim ** power
            if sim >= threshold:
                G.add_edge(matched_idx[i],
                           matched_idx[j_local],
                           weight=sim)

    # 6) Extract components and their mean vectors
    component_means = []
    for comp in nx.connected_components(G):
        vectors = corpus_embeddings[list(comp)]
        component_means.append(np.mean(vectors, axis=0))

    return component_means, nx.number_connected_components(G)
    
def aggregate_tokens(corpus_embedding_matrix,corpus_token_matrix, cosine_thresh:float=float(0.5)):
    '''This uses the Louvain algorithm to aggregate highly similar words, whilst keeping more different 
    words as separate tokens in our matrix. 
    Input:
    corpus_embedding_matrix: the embedding matrix of contextual vectors, for each token
    corpus_token_matrix: the tokens at corresponsing indices
    pwr: raises the edge weights to a power, to increase community separation
    
    Output:
    The aggregated embedding matrix and new corresponding token indices
    community_ns: The number of communities for each word'''

    unique_tokens=list(set(corpus_token_matrix))
    aggregated_embedding_matrix=[]
    aggregated_token_matrix=[]
    community_ns=[]
    for token in tqdm(unique_tokens, desc="Processing tokens"):    
        agg_vecs,num_comms=get_components(token, corpus_embedding_matrix,\
                                          corpus_token_matrix, threshold=cosine_thresh)
        if not agg_vecs:
            continue
        community_ns.append(num_comms)
        aggregated_embedding_matrix.extend(agg_vecs)
        aggregated_token_matrix.extend([token for t in range(0,len(agg_vecs))])
    return np.array(aggregated_embedding_matrix), aggregated_token_matrix,community_ns

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
def clean_mats(embedding_matrix, all_tokens,unique_tokens):

    def remove_char(string):
        return string.replace("Ġ", "")
    def is_english_word(text):
        if text.lower() in unique_tokens:
            return True
        else:
            return False
    
    stop_words = set(stopwords.words('english'))
    unwanted_tokens = {'[CLS]', '[SEP]', '[PAD]', '<pad>','<s>', '</s>','âĢ¦','Ġ','ĠâĢ¦', None}

    def is_valid_token(token):
        if token is None:
            return False
        if token in unwanted_tokens:
            return False
        if token.lower() in stop_words:
            return False
        if '#' in token:
            return False
        if not any(char.isalnum() for char in token):
            return False
        return True
    all_tokens=[remove_char(t) for t in all_tokens]
    
    mask1 = np.array([is_english_word(token) for token in all_tokens])
    filtered_embedding_matrix1 = embedding_matrix[mask1]
    filtered_tokens1 = np.array(all_tokens)[mask1]
    
    mask = np.array([is_valid_token(token) for token in filtered_tokens1])
    filtered_embedding_matrix = filtered_embedding_matrix1[mask]
    filtered_tokens = filtered_tokens1[mask]
    print(list(set(filtered_tokens)))

    return filtered_embedding_matrix, filtered_tokens

def aggregate_embedding_mats(save_dir:str, unique_tokens:list,csv_name:str, window_size:int, overlap:float=float(0.5),\
                              cosine_thresh=int(0.5)):
    toke_save_dir=save_dir+f'CDATM_embeddings/'
    embedding_dir=toke_save_dir+f'{csv_name}_{window_size}_CDATM_embeddings/'

    step = int(window_size * overlap)
    os.listdir(embedding_dir)
    pattern_matrix = rf"^sentence_embedding_matrix_{window_size}_{step}_(\d+)\.pkl$"
    pattern_tokens = rf"^tokens_{window_size}_{step}_(\d+)\.pkl$"

    # Lists to store filenames with captured indices
    matrix_files = []
    token_files = []

    # Populate lists with matching filenames and their indices
    for filename in os.listdir(embedding_dir):
        match_matrix = re.match(pattern_matrix, filename)
        match_tokens = re.match(pattern_tokens, filename)

        if match_matrix:
            index = int(match_matrix.group(1))
            matrix_files.append((index, filename))
        elif match_tokens:
            index = int(match_tokens.group(1))
            token_files.append((index, filename))

    # Sort files by the extracted index in ascending order
    matrix_files.sort()
    token_files.sort()

    # Lists to store loaded data
    corpus_embedding_matrices = []
    corpus_tokens = []

    # Load files in sorted order
    for _, filename in matrix_files:
        with open(os.path.join(embedding_dir, filename), 'rb') as f:
            corpus_embedding_matrices.append(pickle.load(f))

    for _, filename in token_files:
        with open(os.path.join(embedding_dir, filename), 'rb') as f:
            corpus_tokens.append(pickle.load(f))
    corpus_embedding_matrix = np.concatenate(corpus_embedding_matrices, axis=0)
    corpus_token_matrix = np.concatenate(corpus_tokens, axis=0)
    corpus_embedding_matrix, corpus_token_matrix = remove_unwanted_tokens(corpus_embedding_matrix, corpus_token_matrix)


    corpus_embedding_matrix, corpus_token_matrix = clean_mats(corpus_embedding_matrix,\
                                                          corpus_token_matrix,unique_tokens)
                                                          
    aggregated_embedding_mat, aggregated_token_mat,community_numbers= aggregate_tokens(corpus_embedding_matrix,\
                                                                            corpus_token_matrix,cosine_thresh=cosine_thresh)
    save_embedding_dir=f'{save_dir}{csv_name}_aggregated_embedding_mats/'
    save_toke_dir=f'{save_dir}{csv_name}_aggregated_token_mats/'
    os.makedirs(save_embedding_dir, exist_ok=True)
    os.makedirs(save_toke_dir, exist_ok=True)
    save_path = os.path.join(save_embedding_dir, f'{window_size}_{step}_{cosine_thresh}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(aggregated_embedding_mat, f)
        
    save_path_toke = os.path.join(save_toke_dir, f'{window_size}_{step}_{cosine_thresh}.pkl')
    with open(save_path_toke, 'wb') as f:
        pickle.dump(aggregated_token_mat, f)






# In[ ]:




