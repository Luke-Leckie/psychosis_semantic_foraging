#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import os
import random
import numpy as np
random.seed(42)
import sys
import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
import re
import shutil
import re
import pandas as pd
import numpy as np

# In[2]:

def split_text_into_windows(text, window_size, step):
    text = re.sub(r'[.!?,]', '', text)  # This removes full stops, exclamation marks, and question marks
    words = text.split()
    # Create n-sized windows of words
    windows = [' '.join(words[i:i+window_size]) for i in range(0, len(words) - window_size + 1, \
                                                               step)]
    if len(words) % window_size != 0:
        windows.append(' '.join(words[-window_size:]))
    return windows

def process_into_windows(preprocessed_data, window_size=int(20), step=int(10)):
    windows=[]
    for t in preprocessed_data:
        windows.extend(split_text_into_windows(t, window_size, step))
    return windows

def get_contextual_embeddings_and_tokens(save_dir,model,tokenizer, chunks, W_size=20,S_size=10, batch_size=16):
    embedding_matrix = []
    all_tokens = []
    chunks_processed=0
    for i in tqdm(range(0, len(chunks), batch_size)):
        # Process each batch of chunks
        batch_chunks = chunks[i:i + batch_size]
        
        # Tokenize in a single batch to save memory
        tokens = tokenizer(batch_chunks, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids']
        token_words = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        
        # Extend tokens
        all_tokens.extend([token for sublist in token_words for token in sublist])
        
        # Run model inference without storing intermediate activations
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=tokens['attention_mask'], output_hidden_states=True)
            last_layer_embeddings = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
        
        # Detach and convert to numpy arrays directly, instead of creating a list of numpy arrays
        batch_embeddings = last_layer_embeddings.cpu().numpy()
        
        # Flatten batch_embeddings across all chunks and tokens for memory efficiency
        embedding_matrix.extend(batch_embeddings.reshape(-1, batch_embeddings.shape[-1]))
        chunks_processed+=1
        if chunks_processed%500==0:
            with open(f'{save_dir}sentence_embedding_matrix_{W_size}_{S_size}_{chunks_processed}.pkl', 'wb') as f:
                pickle.dump(np.array(embedding_matrix), f)
            with open(f'{save_dir}tokens_{W_size}_{S_size}_{chunks_processed}.pkl', 'wb') as f:
                pickle.dump(np.array(all_tokens), f)
            embedding_matrix = []
            all_tokens = []
    with open(f'{save_dir}sentence_embedding_matrix_{W_size}_{S_size}_{chunks_processed}.pkl', 'wb') as f:
        pickle.dump(np.array(embedding_matrix), f)
    with open(f'{save_dir}tokens_{W_size}_{S_size}_{chunks_processed}.pkl', 'wb') as f:
        pickle.dump(np.array(all_tokens), f)
        

def CDATM_embeddings(df:pd.DataFrame, model_dir:str, save_dir:str,csv_name:str,window:float,atom_step:float=float(0.5)):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    # Ensure the model is in evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    toke_save_dir=save_dir+f'CDATM_embeddings/'
    os.makedirs(toke_save_dir, exist_ok=True)
    preprocessed_data=list(df.processed_text)
    preprocessed_data = list(filter(lambda x: isinstance(x, str), preprocessed_data))

    all_preprocessed_data=process_into_windows(preprocessed_data,window_size=window,step=int(window*atom_step))
    save_dir=toke_save_dir+f'{csv_name}_{window}_CDATM_embeddings/'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    get_contextual_embeddings_and_tokens(save_dir=save_dir,model=model, tokenizer=tokenizer, chunks=all_preprocessed_data, W_size=window, S_size=int(window*atom_step), batch_size=32)


# In[ ]:




