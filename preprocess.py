import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
import gensim.downloader as api
import ast
import torch 

SEQUENCE_LENGTH = 100 
DIAGNOSIS_DIM = 32 
DRUG_DIM = 165
LABS_DIM = 46 
STATIC_DIM = 3 
OBSERVATION_WINDOW_MONTHS = 6

df_demographics = pd.read_csv("./data/final_demographic_death.csv")
df_lab = pd.read_csv("./data/final_lab.csv")
df_opd = pd.read_csv("./data/final_opd_record.csv")
df_drug = pd.read_csv("./data/final_drug.csv") 

# datetime convertion, for datetime calculation
df_demographics['INDATE'] = pd.to_datetime(df_demographics['INDATE'])
df_lab['LOGDATE'] = pd.to_datetime(df_lab['LOGDATE'])
df_opd['CREATEDATETIME'] = pd.to_datetime(df_opd['CREATEDATETIME']) 
df_drug['CREATEDATE'] = pd.to_datetime(df_drug['CREATEDATE'])

def standardize(value, mean, std):
    return (value - mean) / std if std > 0 else value

def process_single_observation(obs_row):

    patient_id = obs_row['PERSONID2']
    admission_date = obs_row['INDATE'] + obs_row['adm_days']    # outdate
    
    # 6 months before last leaving hospital
    start_date = admission_date - pd.DateOffset(months=OBSERVATION_WINDOW_MONTHS)
    
    # -------------- Static Data ---------------------
    target_label = obs_row['death_180'] 
    age = obs_row['age']    # may need standarize
    gender_code = 1.0 if obs_row['ADMINISTRATIVESEXCODE'] == 'M' else 0.0
    los_days = obs_row['adm_days']  # may need standarize

    static_vector = np.array([age, gender_code, los_days], dtype=np.float32)

    # ----------------- LSTM Data -------------------------    
    # Lab
    df_patient_lab = df_lab[df_lab['PERSONID2'] == patient_id]
    df_lab_history = df_patient_lab[
        (df_patient_lab['LOGDATE'] >= start_date) & 
        (df_patient_lab['LOGDATE'] < admission_date)
    ].sort_values('LOGDATE', ascending=False)
    
    # Diagnosis records
    df_patient_opd = df_opd[df_opd['PERSONID2'] == patient_id]
    df_opd_history = df_patient_opd[
        (df_patient_opd['CREATEDATETIME'] >= start_date) & 
        (df_patient_opd['CREATEDATETIME'] < admission_date)
    ].sort_values('CREATEDATETIME', ascending=False)

    # Drug records
    df_patient_drug = df_drug[df_drug['PERSONID2'] == patient_id]
    df_drug_history = df_patient_drug[
        (df_patient_drug['CREATEDATE'] >= start_date) & 
        (df_patient_drug['CREATEDATE'] < admission_date)
    ].sort_values('CREATEDATE', ascending=False)
    

    # -------------- MAKE TENSOR ------------------
    
    # LAB
    lab_cols = [c for c in df_lab.columns if c not in ['PERSONID2', 'LOGDATE']]
    lab_sequence_data = df_lab_history[lab_cols].head(SEQUENCE_LENGTH).values   # truncated to SEQUENCE_LENGTH
    
    lab_tensor = np.zeros((SEQUENCE_LENGTH, LABS_DIM), dtype=np.float32)
    if lab_sequence_data.size > 0:
        # Pad with zeros at the beginning if history is short
        if lab_sequence_data.shape[0] < SEQUENCE_LENGTH:
            padding_rows = SEQUENCE_LENGTH - lab_sequence_data.shape[0]
            lab_sequence_data = np.vstack([np.zeros((padding_rows, lab_sequence_data.shape[1])), lab_sequence_data])
        
    # DRUG
    drug_cols = [c for c in df_drug.columns if c not in ['PERSONID2', 'CREATEDATE']]
    drug_sequence_data = df_drug_history[drug_cols].head(SEQUENCE_LENGTH).values

    drug_tensor = np.zeros((SEQUENCE_LENGTH, DRUG_DIM), dtype=np.float32)
    if drug_sequence_data.size > 0:
        if drug_sequence_data.shape[0] < SEQUENCE_LENGTH:
            padding_rows = SEQUENCE_LENGTH - drug_sequence_data.shape[0]
            drug_sequence_data = np.vstack([np.zeros((padding_rows, DRUG_DIM)), drug_sequence_data])
            
    
    # diagnosis
    diagnosis_sequence = []
    
    # Skip-gram Embedding
    # for codes_str in df_opd_history['icd10_codes']:
    #     codes = ast.literal_eval(codes_str) # eval() return value from string
        
    #     for code in codes:
    #         # Look up the 32D vector (uses zero vector if not found)
    #         vector = MOCK_CODE_VECTORS[code]
    #         diagnosis_sequence.append(vector)

    # 5b. Truncate/Pad to 100
    if len(diagnosis_sequence) > SEQUENCE_LENGTH:
        diagnosis_tensor = np.array(diagnosis_sequence[-SEQUENCE_LENGTH:], dtype=np.float32)
    else:
        padding_rows = SEQUENCE_LENGTH - len(diagnosis_sequence)
        padding = np.zeros((padding_rows, DIAGNOSIS_DIM), dtype=np.float32)
        diagnosis_tensor = np.vstack([padding, np.array(diagnosis_sequence, dtype=np.float32)])

    return {
        'drug': drug_tensor,
        'lab': lab_tensor,
        'diagnosis': diagnosis_tensor,
        'static': static_vector,
        'label': target_label
    }

if __name__ == '__main__':
    final_dataset_list = []
    total_observations = len(df_demographics)
    print(f"Starting generation for {total_observations} admissions using a {OBSERVATION_WINDOW_MONTHS}-month lookback...")

    # Iterate over every admission (observation) in the demographic file
    for index, obs_row in df_demographics.iterrows():
        observation_tensors = process_single_observation(obs_row)
        final_dataset_list.append(observation_tensors)

    # Convert to NumPy array
    final_dataset_array = np.array(final_dataset_list)
    np.save('final_processed_ehr_tensors.npy', final_dataset_array)

    print("\n--- Preprocessing Complete ---")
    print(f"Total observations successfully processed and saved: {len(final_dataset_array)}")
    print("The file 'final_processed_ehr_tensors.npy' is ready for the PyTorch DataLoader.")