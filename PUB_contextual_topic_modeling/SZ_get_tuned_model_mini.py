#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division
import pandas as pd
import pickle
import numpy as np
from gensim import corpora, models, similarities #calc all similarities at once, from http://radimrehurek.com/gensim/tut3.html
from random import seed, sample
import random
from tqdm import tqdm
import numpy as np
random.seed(42)
import sys
from sentence_transformers import SentenceTransformer
import torch
#from transformers import AutoTokenizer, AutoModel  # Import AutoTokenizer and AutoModel
import nltk
nltk.download('stopwords')
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader, Sampler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import string
from sentence_transformers.losses import MultipleNegativesRankingLoss
import torch.nn.functional as F
from collections import defaultdict

def split_text_into_windows(text, window_size=80, step=16):
    """
    Splits text into overlapping windows based on word counts.

    Parameters:
        text (str): The input text.
        window_size (int): The number of words per window.
        step (int): The number of words to shift for the next window.
    
    Returns:
        list: A list of text windows (each a string of words).
    """
    # Split text by whitespace to get words
    try:
        words = text.split()
    except AttributeError:
        return None
    windows = []
    for i in range(0, len(words) - window_size + 1, step):
        window_words = words[i : i + window_size]
        windows.append(" ".join(window_words))
    
    if len(words) % step != 0:
        final_window = " ".join(words[-window_size:])
        if final_window not in windows:
            windows.append(final_window)
    
    return windows

def get_sentence_embeddings(sentences,sentence_model, batch_size=32):
    # If sentences is None or empty, return an empty list
    if not sentences:
        return []
    return sentence_model.encode(sentences, batch_size=batch_size, show_progress_bar=False)

def process_texts_batch(df, sentence_model,text_col='processed_text', windowsize=80, step=16, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_texts = df[text_col].iloc[i:i + batch_size]
        batch_sentences = []
        sentences_count = []
        for text in batch_texts:
            if text is None:
                sentences=None
            else:
                sentences = split_text_into_windows(text, window_size=windowsize, step=step)
            # If the function returns None, replace with an empty list
            if sentences is None:
                sentences = []
            batch_sentences.extend(sentences)
            sentences_count.append(len(sentences))
        # Encode all sentences in one call
        batch_sentence_embeddings = get_sentence_embeddings(batch_sentences, sentence_model,batch_size=batch_size)
        # Reassemble embeddings per text
        idx = 0
        for count in sentences_count:
            all_embeddings.append(batch_sentence_embeddings[idx:idx+count])
            idx += count
    df['sentence_embeddings'] = all_embeddings
    return df


def get_tuned_model(TAD_all:pd.DataFrame,csv_name:str,root_dir:str,save_dir:str,model_dir:str,dir_array:str,window:int, overlap:int, tuning:str):
    step_to_train=int(overlap*window)

    special_tokens = ['<place>','<date>','<institution>','<hospital>',
                    '<place_of_work>','<person_name>','<xxx>']

    TAD_all['windows_to_train'] = TAD_all.apply(lambda row: split_text_into_windows(row['processed_text'], window_size=window,step=step_to_train), axis=1)

    # 5) (Optional) Create separate train/val DataFrames
    TAD_train = TAD_all[TAD_all['split'] == 'train'].reset_index(drop=True)
    TAD_val   = TAD_all[TAD_all['split'] == 'val'].reset_index(drop=True)

    training_windows = []
    window_to_doc_id = {}  # Maps each window text to its source document

    for doc_idx, window_list in enumerate(TAD_all['windows_to_train']):
        for window_text in window_list:
            # Filter windows during creation, not after
            if isinstance(window_text, str) and len(window_text.split()) >= window * 0.8:
                training_windows.append(window_text)
                window_to_doc_id[window_text] = doc_idx


    print('training data size: ', len(training_windows))
    # 1) Load transformer backbone
    
    # Example special tokens
    special_tokens = ['<place>', '<date>', '<institution>', '<hospital>', '<place_of_work>', '<person_name>', '<xxx>']

    # Load model
    model_name = root_dir + 'Models/all-MiniLM-L6-v2'
    word_embedding_model = models.Transformer(model_name)

    # Add special tokens to tokenizer
    tokenizer = word_embedding_model.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    word_embedding_model.tokenizer = tokenizer

    # Get embedding layer
    embedding_layer = word_embedding_model.auto_model.embeddings.word_embeddings
    tokenizer = word_embedding_model.tokenizer

    # Define reference tokens
    male_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"]
    female_names = ["Mary", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer", "Maria", "Susan", "Margaret", "Dorothy"]
    all_names = male_names + female_names

    # Generate 10 random dates in "DD-MM-YYYY" format
    def random_date():
        day = str(random.randint(1, 28)).zfill(2)   # Avoid invalid days
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1900, 2025))
        return f"{day}-{month}-{year}"

    date_tokens = [random_date() for _ in range(10)]

    # Mapping from special token to reference tokens
    init_map = {
        '<place>': ['city'],
        '<date>': date_tokens,  # List of 10 random dates
        '<institution>': ['college', 'school'],
        '<hospital>': ['hospital'],
        '<place_of_work>': ['workplace'],
        '<person_name>': all_names,  # 20 real names
        '<xxx>': None  # Leave random
    }

    # Get the index of each special token
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    # Initialize embeddings
    with torch.no_grad():
        for token, token_id in zip(special_tokens, special_token_ids):
            if init_map[token] is None:
                continue  # Leave random
            ref_tokens = init_map[token]
            # Convert each reference token to ids
            ref_ids = []
            for t in ref_tokens:
                t_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))
                ref_ids.extend(t_ids)
            # Average embeddings of reference tokens
            ref_embeds = embedding_layer.weight[ref_ids]
            avg_embed = ref_embeds.mean(dim=0)
            embedding_layer.weight[token_id] = avg_embed.clone()

    for param in word_embedding_model.auto_model.embeddings.parameters():
        param.requires_grad = False
    freeze_layers = 0
    for layer in word_embedding_model.auto_model.encoder.layer[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    # 3) Add pooling
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_examples = []
    doc_windows = {}

    # Group windows by document ID
    for window_text in training_windows:
        doc_id = window_to_doc_id[window_text]
        if doc_id not in doc_windows:
            doc_windows[doc_id] = []
        doc_windows[doc_id].append(window_text)

    print(f"Number of documents: {len(doc_windows)}")
    print(f"Average windows per document: {len(training_windows) / len(doc_windows):.1f}")
    class WeightedMNRLoss(MultipleNegativesRankingLoss):
        def __init__(self, model, alpha: float = 0.1):
            """
            model: your SentenceTransformer
            alpha: controls how sharply the weight decays with distance
            weight = exp(-alpha * distance)
            """
            super().__init__(model)   # this sets self.model, self.scale, self.similarity_fct
            self.alpha = alpha

        def forward(self, sentence_features, labels):
            # 1) embed each “column” of features
            #    sentence_features is an iterable of dicts, one dict per input (anchor, positive)
            reps = [ self.model(feat_dict)["sentence_embedding"]
                    for feat_dict in sentence_features ]
            # Reps[0]: (batch_size, dim), Reps[1]: (batch_size, dim)
            anchors   = reps[0]
            positives = torch.cat(reps[1:], dim=0)  # same as reps[1] if you only passed two

            # 2) compute similarity scores just like MNLoss does
            #    dot-product or cosine ‒ here we’ll mimic MNLoss’s default:
            scores = self.similarity_fct(anchors, positives) * self.scale
            # scores: (batch_size, batch_size) — each anchor vs. every positive in the batch

            # 3) get per-example info-nce loss
            log_prob = F.log_softmax(scores, dim=1)
            batch_size = anchors.size(0)
            idx = torch.arange(batch_size, device=scores.device)
            per_example_loss = -log_prob[idx, idx]   # (batch_size,)

            # 4) build your distance weights
            distances = labels.to(scores.device)     # (batch_size,)
            weights   = 1.0 - 0.25 * (distances - 1)  
            # 5) weighted average
            loss = (weights * per_example_loss).sum() / weights.sum()
            return loss

    # -----------------------------
    # Build your examples (storing distance in `label`)
    train_examples = []
    for doc_id, windows in doc_windows.items():
        i=0
        while i < len(windows):
            if i + 1 < len(windows):
                train_examples.append(
                    InputExample(texts=[windows[i], windows[i+1]],
                                label=1.0)
                )
            # medium/hard positives
            for j in [i+random.choice([2,3,4])]:
                if j < len(windows):
                    train_examples.append(
                        InputExample(texts=[windows[i], windows[j]],
                                    label=float(j - i))
                    )
            step = 10
            i += step 
    print(f'Total training examples: {len(train_examples)}')
    #from sentence_transformers.losses import SentenceTransformerLoss
    train_loss = WeightedMNRLoss(model, alpha=0.2)
    print(f'Total training examples: {len(train_examples)}')
    # train_dataloader = DataLoader(
    #     train_examples,
    #     batch_sampler=DocBatchSampler(train_examples, 128),
    #     # default collate will just return a list of InputExample
    # )
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)

    # # Create validation set
    # val_examples = train_examples[:100]  # Sample for validation
    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="validation")
    # 7) Fine-tune
    num_epochs   = 3
    warmup_ratio = 0.1
    total_steps  = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    model.old_fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_dir,
        show_progress_bar=True
    )
    # Clear training data after fine-tuning
    del training_windows, train_examples, train_dataloader

def get_tuned_window_embeddings(TAD_all:pd.DataFrame,csv_name:str,root_dir:str,save_dir:str,model_dir:str,dir_array:str,window:int, overlap:int, tuning:str):
    torch.cuda.empty_cache()  # If using GPU
    model = SentenceTransformer(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    overlap=0.1
    step=1#int(window*overlap)#4data_name_list
    print('data size: ',len(TAD_all))
    print('window: ', window)
    df_processed = process_texts_batch(TAD_all, sentence_model=model,\
                                    windowsize=window,step=step,batch_size=16)
    print(len(df_processed['sentence_embeddings']))
    with open(f'{dir_array}{window}_{1}_{csv_name}_wind_embeddings_{tuning}.pkl', 'wb') as f:
        pickle.dump(list(df_processed['sentence_embeddings']), f)


