# import MV_get_noun_embeddings as MV_noun
import SZ_get_tuned_model_mini as SZ_window
import SZ_get_DATM_embeddings as SZ_CDATM_w
import SZ_aggregate_embedding_mats as SZ_agg
import SZ_CDATM as SZ_CDATM
import pickle
import pandas as pd
import sys
import os
import numpy as np

def is_long_str(x, min_words=100):
    return isinstance(x, str) and len(x.split()) >= min_words

usr='cluster'
test_mode=False
if usr=='luke':
    working_dir='/home/ll16598/Documents/POSTDOC/SCHIZ/IPII_pipe/SZ_tm_cluster_tuned/'
    root_dir='/home/ll16598/Documents/POSTDOC/SCHIZ/IPII_pipe/SZ_tm_cluster_tuned/SZ_get_tuned_contextual_embeddings/'

    csv_name='IPIIs_censored'
    window=20
    atom_step=0.5
    tau=0.5
    RM_C0=False
elif usr=='cluster':
    working_dir='/N/u/lleckie/Quartz/work/SZ_tm_cluster_tuned/'
    root_dir='/N/u/lleckie/Quartz/work/SZ_tm_cluster_tuned/SZ_get_tuned_contextual_embeddings/'

    csv_name='IPIIs_censored'
    window=int(sys.argv[1])
    tau=float(sys.argv[2])
    atom_step=float(sys.argv[3])
    RM_C0=False#sys.argv[5]

overlap=0.2
step_to_train=int(overlap*window)
step=1
data_column_name='processed_text'
data_dir=working_dir+f'{csv_name}.csv'
data_dir_lab=working_dir+f'{csv_name}_train_labelled.csv'

save_dir=working_dir+'SZ_derived_data/'
if usr=='cluster':
    save_dir='/N/scratch/lleckie/SZ_derived_data/'
array_type='mini_01'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(root_dir, exist_ok=True)

dir_array=save_dir+'SZ_window_vectors/'

os.makedirs(dir_array, exist_ok=True)

model_dir=f'{root_dir}{window}_{step_to_train}_SZ-mini-finetuned_{array_type}'
df = pd.read_csv(data_dir)
print(list(df.columns))
#mask = df['processed_text'].apply(is_long_str)
#df = df[mask].reset_index(drop=True)

if test_mode:
    df=df[0:1]

for idx, t in enumerate(list(df[data_column_name])):
    print(idx, t[:50],'.........', t[-50:])


df['split'] = 'train'
val_idx = df.sample(frac=0.1, random_state=42).index
# 4) Mark those rows as "val"
df.loc[val_idx, 'split'] = 'val'
df.to_csv(data_dir_lab)
#if window in [40]:
print('ROOT:', root_dir)
SZ_window.get_tuned_model(df,overlap=overlap,tuning=array_type,csv_name=csv_name,root_dir=root_dir,model_dir=model_dir,dir_array=dir_array, save_dir=save_dir,window=window)
print('Completed step 2, window embeddings')
SZ_window.get_tuned_window_embeddings(df,overlap=overlap,tuning=array_type,csv_name=csv_name,root_dir=root_dir,model_dir=model_dir,dir_array=dir_array, save_dir=save_dir,window=window)
SZ_CDATM_w.CDATM_embeddings(df, model_dir=model_dir,save_dir=save_dir,window=window,csv_name=csv_name)
print('Completed step 3, CDATM embeddings')
SZ_agg.aggregate_embedding_mats(save_dir=save_dir,csv_name=csv_name,window_size=window, overlap=atom_step, cosine_thresh=tau)
print('Completed step 4, aggregate embedding matrix')
SZ_CDATM.model_corpus(df=df,save_dir=save_dir,array_type=array_type,csv_name=csv_name,window=window, atom_step=atom_step, cosine_thresh=tau)
#SZ_CDATM.model_corpus(df=df,save_dir=save_dir,array_type=array_type,csv_name=csv_name,window=window, atom_step=atom_step, cosine_thresh=tau,RM_C0=False)
print('Completed step 4, build and apply topic models')
