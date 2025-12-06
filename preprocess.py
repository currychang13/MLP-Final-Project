import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
from tqdm import tqdm
import os

SEQUENCE_LENGTH = 100 
DIAGNOSIS_DIM = 32 
DRUG_DIM = 165
LABS_DIM = 46 
STATIC_DIM = 3 
OBSERVATION_WINDOW_MONTHS = 6
MODEL_FILE = './skip_gram/diagnosis_skipgram.model'

print("Loading CSV files...")
os.makedirs("./data", exist_ok=True)
df_demographics = pd.read_csv("./data/final_demographic_death.csv")
df_lab = pd.read_csv("./data/final_lab.csv")
df_opd = pd.read_csv("./data/final_opd_record.csv")
df_drug = pd.read_csv("./data/final_drug.csv") 

# datetime convertion, for datetime calculation
df_demographics['INDATE'] = pd.to_datetime(df_demographics['INDATE'])
df_lab['LOGDATE'] = pd.to_datetime(df_lab['LOGDATE'])
df_opd['CREATEDATETIME'] = pd.to_datetime(df_opd['CREATEDATETIME']) 
df_drug['CREATEDATE'] = pd.to_datetime(df_drug['CREATEDATE'])

# --- Patient-Level Split (80-10-10) ---
print("Performing Patient-Level Train/Val/Test Split...")
unique_patient_ids = df_demographics['PERSONID2'].unique()
np.random.seed(42) 
np.random.shuffle(unique_patient_ids)

n_patients = len(unique_patient_ids)
n_train = int(n_patients * 0.80)
n_val = int(n_patients * 0.10)

train_pids = set(unique_patient_ids[:n_train])
val_pids = set(unique_patient_ids[n_train : n_train + n_val])
test_pids = set(unique_patient_ids[n_train + n_val:])

print(f"Total Patients: {n_patients}")
print(f"Train Patients: {len(train_pids)}")
print(f"Val Patients:   {len(val_pids)}")
print(f"Test Patients:  {len(test_pids)}")


def standardize(value, mean, std):
    return (value - mean) / std if std > 0 else 0

def process_single_observation(obs_row, word_vectors, lab_stats, static_stats):

    patient_id = obs_row['PERSONID2']
    admission_date = obs_row['INDATE'] + pd.to_timedelta(obs_row['adm_days'], unit='D')    # leave date
    
    # 6 months before last leaving hospital
    start_date = admission_date - pd.DateOffset(months=OBSERVATION_WINDOW_MONTHS)


    # ------------- STATIC DATA FILTERING ---------------
    target_label = np.array([obs_row['death_180'], obs_row['death_30']], dtype=np.float32)     # Multi-label 

    gender_code = 1.0 if obs_row['ADMINISTRATIVESEXCODE'] == 'M' else 0.0

    # standardization
    age = standardize(obs_row['age'], static_stats['age']['mean'], static_stats['age']['std'])
    los_days = standardize(obs_row['adm_days'], static_stats['los']['mean'], static_stats['los']['std'])

    static_vector = np.array([age, gender_code, los_days], dtype=np.float32)

    # ----------------- LSTM Data filtering -------------------------    
    df_patient_lab = df_lab[df_lab['PERSONID2'] == patient_id]
    df_lab_history = df_patient_lab[
        (df_patient_lab['LOGDATE'] >= start_date) & 
        (df_patient_lab['LOGDATE'] < admission_date)
    ].sort_values('LOGDATE', ascending=False)
    
    df_patient_opd = df_opd[df_opd['PERSONID2'] == patient_id]
    df_opd_history = df_patient_opd[
        (df_patient_opd['CREATEDATETIME'] >= start_date) & 
        (df_patient_opd['CREATEDATETIME'] < admission_date)
    ].sort_values('CREATEDATETIME', ascending=False)

    df_patient_drug = df_drug[df_drug['PERSONID2'] == patient_id]
    df_drug_history = df_patient_drug[
        (df_patient_drug['CREATEDATE'] >= start_date) & 
        (df_patient_drug['CREATEDATE'] < admission_date)
    ].sort_values('CREATEDATE', ascending=False)
    

    # -------------- MAKE TENSOR ------------------
    # 1. Padding for consistent sequence length
    # 2. Concate to fixed sequence length
    
    # ------------------- lab --------------------
    lab_cols = [c for c in df_lab.columns if c not in ['PERSONID2', 'LOGDATE', '%CKMB']]
    lab_sequence_data = df_lab_history[lab_cols].head(SEQUENCE_LENGTH).values   # truncated to SEQUENCE_LENGTH
    lab_len = lab_sequence_data.shape[0]

    lab_tensor = np.zeros((SEQUENCE_LENGTH, LABS_DIM), dtype=np.float32)
    if lab_sequence_data.size > 0:
        
        # Standardize
        for i, col_name in enumerate(lab_cols):
            mu = lab_stats[col_name]['mean']
            sigma = lab_stats[col_name]['std']
            
            if sigma > 0:
                lab_sequence_data[:, i] = (lab_sequence_data[:, i] - mu) / sigma
            else:
                lab_sequence_data[:, i] = 0

        # post padding
        if lab_sequence_data.shape[0] < SEQUENCE_LENGTH:
            padding_rows = SEQUENCE_LENGTH - lab_sequence_data.shape[0]
            lab_sequence_data = np.vstack([lab_sequence_data, np.zeros((padding_rows, lab_sequence_data.shape[1]))])
        
        current_cols = min(LABS_DIM, lab_sequence_data.shape[1])
        lab_tensor[:, :current_cols] = lab_sequence_data[:, :current_cols]
        
    # --------------- DRUG --------------
    drug_cols = [c for c in df_drug.columns if c not in ['PERSONID2', 'CREATEDATE']]
    drug_sequence_data = df_drug_history[drug_cols].head(SEQUENCE_LENGTH).values
    drug_len = len(drug_sequence_data)

    drug_tensor = np.zeros((SEQUENCE_LENGTH, DRUG_DIM), dtype=np.float32)
    if drug_sequence_data.size > 0:
        if drug_sequence_data.shape[0] < SEQUENCE_LENGTH:
            padding_rows = SEQUENCE_LENGTH - drug_sequence_data.shape[0]
            drug_sequence_data = np.vstack([drug_sequence_data, np.zeros((padding_rows, DRUG_DIM))])
            
        current_cols = min(DRUG_DIM, drug_sequence_data.shape[1])
        drug_tensor[:, :current_cols] = drug_sequence_data[:, :current_cols]
    
    # ------------- diagnosis ------------------
    diagnosis_sequence = []

    # Skip-gram Embedding
    for codes_str in df_opd_history['icd10_codes']:
        codes = ast.literal_eval(codes_str) # eval() return value from string
        
        for code in codes:
            if code in word_vectors:
                vector = word_vectors[code]
            else:
                vector = np.zeros(DIAGNOSIS_DIM, dtype=np.float32)
            
            diagnosis_sequence.append(vector)
            print(vector)


    if len(diagnosis_sequence) > 0:
        diagnosis_sequence = np.array(diagnosis_sequence, dtype=np.float32)
    else:
        diagnosis_sequence = np.empty((0, DIAGNOSIS_DIM), dtype=np.float32)

    diagnosis_sequence = diagnosis_sequence[:SEQUENCE_LENGTH].copy()
    diag_len = diagnosis_sequence.shape[0]

    diagnosis_tensor = np.zeros((SEQUENCE_LENGTH, DIAGNOSIS_DIM), dtype=np.float32)
    if diagnosis_sequence.shape[0] < SEQUENCE_LENGTH:
        padding_rows = SEQUENCE_LENGTH - diagnosis_sequence.shape[0]
        padding = np.zeros((padding_rows, DIAGNOSIS_DIM), dtype=np.float32)
        diagnosis_tensor = np.vstack([diagnosis_sequence, padding])

    return {
        'drug': drug_tensor,
        'drug_len': drug_len, 
        'lab': lab_tensor,
        'lab_len': lab_len,   
        'diagnosis': diagnosis_tensor,
        'diag_len': diag_len, 
        'static': static_vector,
        'label': target_label
    }

if __name__ == '__main__':

    print(f"Loading Skip-gram model from {MODEL_FILE}...")
    try:
        w2v_model = Word2Vec.load(MODEL_FILE)
        word_vectors = w2v_model.wv
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_FILE} not found. Please run 'train_skipgram.py' first.")
        exit(1)
    
    train_list = []
    val_list = []
    test_list = []

    total_observations = len(df_demographics)
    print(f"Starting generation for {total_observations} admissions using a {OBSERVATION_WINDOW_MONTHS}-month lookback...")
    
    # Filter the dataframes to get only training data for calculating statistics
    train_demographics = df_demographics[df_demographics['PERSONID2'].isin(train_pids)]
    train_lab = df_lab[df_lab['PERSONID2'].isin(train_pids)]

    static_stats = {
        'age': {
            'mean': train_demographics['age'].mean(),
            'std': train_demographics['age'].std()
        },
        'los': {
            'mean': train_demographics['adm_days'].mean(),
            'std': train_demographics['adm_days'].std()
        }
    }

    lab_cols = [c for c in df_lab.columns if c not in ['PERSONID2', 'LOGDATE', '%CKMB']]
    lab_stats = {}

    for col in lab_cols:
        lab_stats[col] = {
            'mean': train_lab[col].mean(),
            'std': train_lab[col].std()
        }

    # Iterate over every admission (observation) in the demographic file
    for index, obs_row in tqdm(df_demographics.iterrows(), total=len(df_demographics), desc="Preprocessing"):
        pid = obs_row['PERSONID2']
        observation_tensors = process_single_observation(obs_row, word_vectors, lab_stats, static_stats)

        # patient level train/val/test
        if pid in train_pids: train_list.append(observation_tensors)
        elif pid in val_pids: val_list.append(observation_tensors)
        elif pid in test_pids: test_list.append(observation_tensors)

    # Convert to NumPy array
    np.save('train.npy', np.array(train_list))
    np.save('validate.npy', np.array(val_list))
    np.save('test.npy', np.array(test_list))

    print("\n--- Preprocessing Complete ---")
    print(f"Train Observations: {len(train_list)}")
    print(f"Val Observations:   {len(val_list)}")
    print(f"Test Observations:  {len(test_list)}")
    print("Files saved: train.npy, validate.npy, test.npy")