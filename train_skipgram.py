import pandas as pd
import numpy as np
import pickle
import ast
import os
from gensim.models import Word2Vec

# --- Configuration ---
OBSERVATION_WINDOW_MONTHS = 6
DIAGNOSIS_DIM = 32
os.makedirs("./skip_gram", exist_ok=True)
CORPUS_FILE = './skip_gram/diagnosis_corpus.pkl'
MODEL_FILE = './skip_gram/diagnosis_skipgram.model'

def load_and_split_data(training_set_only):
    print("Loading Demographics for split...")
    df_demographics = pd.read_csv("./data/final_demographic_death.csv")
    
    # Unique Patients
    unique_patient_ids = df_demographics['PERSONID2'].unique()
    
    np.random.seed(42) 
    np.random.shuffle(unique_patient_ids)

    n_patients = len(unique_patient_ids)
    n_train = int(n_patients * 0.80)
    
    if training_set_only:
        train_pids = set(unique_patient_ids[:n_train])
    else:
        train_pids = set(unique_patient_ids)
    
    return train_pids, df_demographics

def build_temporal_corpus(train_pids, df_demographics):
    print("Loading OPD records...")
    df_opd = pd.read_csv("./data/final_opd_record.csv")
    
    df_demographics['INDATE'] = pd.to_datetime(df_demographics['INDATE'])
    df_opd['CREATEDATETIME'] = pd.to_datetime(df_opd['CREATEDATETIME'])
    
    print("Building Corpus (Temporal Filtering)...")
    corpus = []
    
    # Only keep records relevant to training patients
    opd_subset = df_opd[df_opd['PERSONID2'].isin(train_pids)].copy()
    opd_grouped = opd_subset.groupby('PERSONID2')
    
    # Iterate only through training patients in demographics
    train_demos = df_demographics[df_demographics['PERSONID2'].isin(train_pids)]
    
    count = 0
    for _, row in train_demos.iterrows():
        pid = row['PERSONID2']
        
        admission_date = row['INDATE'] + pd.to_timedelta(row['adm_days'], unit='D')
        start_date = admission_date - pd.DateOffset(months=OBSERVATION_WINDOW_MONTHS)
        
        if pid in opd_grouped.groups:
            hist = opd_grouped.get_group(pid)
            
            mask = (hist['CREATEDATETIME'] >= start_date) & (hist['CREATEDATETIME'] < admission_date)
            valid_hist = hist[mask].sort_values('CREATEDATETIME')
            
            seq_codes = []
            for codes_str in valid_hist['icd10_codes']:
                try:
                    codes = ast.literal_eval(codes_str)
                    seq_codes.extend(codes)
                except:
                    continue
            
            if seq_codes:
                corpus.append(seq_codes)
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} admissions...")
            
    return corpus

def save_corpus(corpus):
    print(f"Saving corpus with {len(corpus)} sequences to {CORPUS_FILE}...")
    with open(CORPUS_FILE, 'wb') as f:
        pickle.dump(corpus, f)

def train_model(corpus):
    print(f"Training Word2Vec on {len(corpus)} sequences...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=DIAGNOSIS_DIM,
        window=5,      # Context window size
        min_count=1,   # Include rare codes
        sg=1,          # 1 = Skip-gram
        workers=4,
        epochs=20
    )
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    train_pids, df_demos = load_and_split_data(training_set_only=False)
    corpus = build_temporal_corpus(train_pids, df_demos)
    save_corpus(corpus)
    train_model(corpus)