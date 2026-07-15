

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from itertools import combinations



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

    return filtered_embedding_matrix, list(filtered_tokens)
    
    
def apply_aksvd(embedding_matrix, n_components, n_nonzeros,  savelocation,save=True):
    """
    Applies Approximate K-SVD to the contextual embeddings matrix.
    
    Args:
    - embedding_matrix: Matrix of token embeddings (shape: num_tokens x embedding_dim).
    - n_components: Number of dictionary atoms.
    - n_nonzeros: Sparsity level (number of nonzero coefficients).
    
    Returns:
    - dictionary: The learned dictionary.
    - sparse_codes: The sparse codes representing the input data.
    """
    aksvd_t = ApproximateKSVD(n_components=n_components, transform_n_nonzero_coefs=n_nonzeros)
    dictionary_t = aksvd_t.fit(embedding_matrix).components_  # The dictionary
    gamma_t = aksvd_t.transform(embedding_matrix) #get the gammas, which are the "weights" of each word on a discourse atoms
    outfile = open(str(savelocation)  +str(n_components) + 'comp' + str(n_nonzeros) + 'nonzeros_dictionary'+'_','wb')
    pickle.dump(dictionary_t,outfile)
    outfile.close()

    outfile = open(str(savelocation)  + str(n_components) + 'comp' + str(n_nonzeros) + 'nonzeros_gamma'+'_','wb')
    pickle.dump(gamma_t,outfile)
    outfile.close()
    return dictionary_t, gamma_t#aksvd_t.transform(embedding_matrix)  # dictionary, sparse codes

def reconst_qual(gpt2_embeddings, dictionary_mat, gamma_mat):
    """
    Measures the reconstruction quality for the GPT-2 embeddings.
    
    Args:
    - gpt2_embeddings (numpy.ndarray): Embeddings matrix from GPT-2 (shape: num_tokens x embedding_dim).
    - dictionary_mat (numpy.ndarray): Learned dictionary (shape: n_components x embedding_dim).
    - gamma_mat (numpy.ndarray): Sparse codes for the embeddings (shape: num_tokens x n_components).
    
    Returns:
    - sse3: Sum of squared errors.
    - rmse: Root mean square error.
    - r2: R² score.
    """
    # Reconstruct the embeddings
    reconstructed = gamma_mat.dot(dictionary_mat)
    
    # Calculate total variance (SST)
    squares3 = gpt2_embeddings - np.mean(gpt2_embeddings, axis=1).reshape(-1, 1)
    sst3 = np.sum(np.square(squares3))
    
    # Calculate sum of squared errors (SSE)
    e3 = reconstructed - gpt2_embeddings
    sse3 = np.sum(np.square(e3))
    
    # Calculate R²
    r2 = 1 - (sse3 / sst3)
    
    # Calculate RMSE
    rmse = math.sqrt(np.mean(np.square(e3)))
    
    return sse3, rmse, r2

def topic_diversity(gpt2_embeddings, dictionary_mat, tokens, top_n=25):
    """
    Measures the diversity of the top unique words within each topic based on cosine similarity.
    
    Args:
    - gpt2_embeddings (numpy.ndarray): Embeddings matrix from GPT-2 (shape: num_tokens x embedding_dim).
    - dictionary_mat (numpy.ndarray): Learned dictionary (shape: n_components x embedding_dim).
    - tokens (list): List of tokens corresponding to the rows in the embedding matrix.
    - top_n (int): Number of top unique words to consider.
    
    Returns:
    - diversity: Diversity score of the topics.
    """
    
    topwords = []

    # Find the top_n most similar unique words for each topic
    for i in range(len(dictionary_mat)):
        
        # Compute cosine similarities between the dictionary atom and all tokens
        similarities = cosine_similarity(dictionary_mat[i].reshape(1, -1), gpt2_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        top_sorted_indices=sorted_indices[0:25]
        topwords.extend([tokens[idx] for idx in top_sorted_indices])
    uniquewords = set(topwords)
    diversity = len(uniquewords) / len(topwords)
    
    return diversity

def coherence_pairwise(gpt2_embeddings, dictionary_mat, tokens, top_n=25):
    """
    Measures pairwise coherence of the top unique words in each topic based on cosine similarity.
    
    Args:
    - gpt2_embeddings (numpy.ndarray): Embeddings matrix from GPT-2 (shape: num_tokens x embedding_dim).
    - dictionary_mat (numpy.ndarray): Learned dictionary (shape: n_components x embedding_dim).
    - tokens (list): List of tokens corresponding to the rows in the embedding matrix.
    - top_n (int): Number of top unique words to consider.
    
    Returns:
    - mean_coherence: Average pairwise coherence score.
    """
    meansim = []

    # For each dictionary atom (topic)
    for k in dictionary_mat:
        # Get the cosine similarities for each token embedding relative to the topic vector
        similarities = cosine_similarity(k.reshape(1, -1), gpt2_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]  # Sort by similarity in descending order
        top_sorted_indices=sorted_indices[0:25]
        topwordsvecs = gpt2_embeddings[top_sorted_indices]
        combo_sims = [
            abs(cosine_similarity(l[0].reshape(1, -1), l[1].reshape(1, -1))[0][0])
            for l in combinations(topwordsvecs, 2)
        ]
        meansim.append(np.mean(combo_sims))  # Coherence score for this topic

    return np.mean(meansim)


# In[ ]:
def remove_c0(comdiscvec, modcontextvecs):
    curcontextvec= [X - X.dot(comdiscvec.transpose()) * comdiscvec for X in modcontextvecs] #remove c_0 from all the cts
    curcontextvec=np.asarray(curcontextvec)
    return(curcontextvec)

def get_most_similar_atoms(sentence_embeddings, atom_vectors,c0,co_remove=False):
    most_similar_atoms = []
    most_sims=[]
    if co_remove:
        sentence_embeddings=remove_c0(c0,sentence_embeddings)
    for embedding in sentence_embeddings:
        embedding_reshaped = embedding.reshape(1, -1)
        similarities = np.array([cosine_similarity(embedding_reshaped, av.reshape(1, -1)) for av in atom_vectors])
        #most_similar_atom_idx = np.argmax(similarities)
        similarities=[i[0][0] for i in similarities]
        sorted_idx = np.argsort(similarities)
        sorted_sims=sorted(similarities)
        top4_idx = sorted_idx[-4:][::-1]
      #  most_similar_atoms.append(most_similar_atom_idx)
        most_similar_atoms.append(list(top4_idx))
        most_sims.append(sorted_sims[-4:][::-1])
    return most_similar_atoms, most_sims

def process_texts_batch(df,atom_dict, c0,batch_size=32,co_rem=False):

    # Columns to store results
    most_similar_atoms_column = []
    most_similar_atoms_sims_column = []

    # Wrapping the loop with tqdm for progress tracking over batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        # Process each batch
        batch_texts = df['sentence_embeddings'].iloc[i:i + batch_size]

        batch_most_similar_atoms_no_cluster=[]
        batch_most_sims=[]
        # Iterate over texts in the current batch
        for sentence_embeddings in batch_texts:
            most_similar_atoms_no_cluster, top_sims = get_most_similar_atoms(sentence_embeddings, atom_dict,c0=c0,co_remove=co_rem)
            batch_most_similar_atoms_no_cluster.append(most_similar_atoms_no_cluster)
            batch_most_sims.append(top_sims)


        # Extend the main columns with batch results
        most_similar_atoms_column.extend(batch_most_similar_atoms_no_cluster)
        most_similar_atoms_sims_column.extend(batch_most_sims)

    # Add results to the dataframe
    df['atom_seq'] = most_similar_atoms_column
    df['atom_seq_cos'] = most_similar_atoms_sims_column

    return df

import re
def tokenize_text(text):
    if pd.isna(text):
        return []
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return text.split()
    
def get_c0(sampvecs):
    svd = TruncatedSVD(n_components=1, n_iter=10, random_state=0) #only keeping top component, using same method as in SIF embedding code
    svd.fit(sampvecs) #1st singular vector  is now c_o
    return(svd.components_[0])  



def model_corpus(df:pd.DataFrame, save_dir:str,array_type:str,window:int,csv_name:str,atom_step:float, cosine_thresh:float):
        

    step = int(window * atom_step)

    dir_array=save_dir+'SZ_window_vectors/'
    with open(f'{dir_array}{window}_{1}_{csv_name}_wind_embeddings_{array_type}.pkl', 'rb') as f:
        embeds=pickle.load(f)
    df['sentence_embeddings']= embeds
    # Get all unique tokens
    global_embeddings=[]
    for i in embeds:
        global_embeddings.extend(i)

    dir_save_atoms=save_dir+'topic_assigned_dfs/'
    os.makedirs(dir_save_atoms, exist_ok=True)
    #if RM_C0:
    c0=get_c0(global_embeddings)

    save_embedding_dir=f'{save_dir}{csv_name}_aggregated_embedding_mats/'
    save_toke_dir=f'{save_dir}{csv_name}_aggregated_token_mats/'
    save_path = os.path.join(save_embedding_dir, f'{window}_{step}_{cosine_thresh}.pkl')
    with open(save_path, 'rb') as f:
       aggregated_embedding_mat= pickle.load(f)
        
    save_path_toke = os.path.join(save_toke_dir, f'{window}_{step}_{cosine_thresh}.pkl')
    with open(save_path_toke, 'rb') as f:
       aggregated_token_mat=pickle.load(f)


    dict_save_dir=f'{save_dir}{csv_name}_sBERT_models_{window}_{step}_{cosine_thresh}/'
    os.makedirs(dict_save_dir, exist_ok=True)
    ntopics= []
    nonzeros = []
    cohere_pairwise= []
    div=[]
    sse= []
    rmse =[]
    r2=[]

    for n_nonzeros in [5,2,10]:
        for n_components in [50,75,100,125,150,25]: #25, 50, range(100,300,50)
                dictionary, gamma = apply_aksvd(aggregated_embedding_mat, n_components, n_nonzeros,dict_save_dir)
                cohere_pairwise.append(coherence_pairwise(aggregated_embedding_mat, dictionary, aggregated_token_mat, \
                                                        top_n=25))
                div.append(topic_diversity(aggregated_embedding_mat, dictionary, aggregated_token_mat, top_n=25))
                rec = reconst_qual(aggregated_embedding_mat, dictionary, gamma)
                sse.append(rec[0])
                rmse.append(rec[1])
                r2.append(rec[2])
                ntopics.append(n_components)
                nonzeros.append(n_nonzeros)
                print(f'{n_components} topics {n_nonzeros} nonzeros, {rec}')
                print(n_components, n_nonzeros)
                for RM_C0 in [False]:    
                    if RM_C0:
                        corm='_C0rm'
                        print('removing global context vector')
                    else:
                        corm=''
                        print('NOT removing global context vector')
                    df_atoms=process_texts_batch(df,dictionary,c0=c0,co_rem=RM_C0)
                    df_atoms.to_csv(dir_save_atoms+f'{csv_name}_atoms_{window}_{n_components}_{n_nonzeros}_{cosine_thresh}{corm}.csv')

        quality_results = pd.DataFrame(data={'Components_Topics': ntopics,'Nonzeros': nonzeros,
                    'CohereCossim_top25_mean': cohere_pairwise, 'Diversity_top25': div, 
                    'SSE': sse,'RMSE': rmse, 'R2': r2})
        quality_results.to_csv(f'{dict_save_dir}{csv_name}_BERT_topicmodel_quality_results_{window}_{atom_step}_{cosine_thresh}.csv')
