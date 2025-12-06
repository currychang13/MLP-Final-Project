import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import EnsembleHFPredictor
from tqdm import tqdm

class EHRDataset(Dataset):
    def __init__(self, processed_data_path):
        self.data = np.load(processed_data_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.data)

    def _process_sequence(self, raw_tensor):
        """
        Converts Pre-Padded tensor (zeros at start) to Post-Padded (zeros at end)
        and returns the actual length.
        """
        # 1. Identify rows that are NOT all zeros
        # raw_tensor shape: (Max_Len, Features)
        non_zero_mask = ~np.all(raw_tensor == 0, axis=1)
        actual_length = np.sum(non_zero_mask)
        
        # Edge Case: If sequence is empty (all zeros), keep length 1 to avoid NaN in LSTM
        if actual_length == 0:
            actual_length = 1
            # Tensor remains all zeros, which is fine
            return torch.from_numpy(raw_tensor).float(), actual_length

        # 2. Extract valid data
        valid_data = raw_tensor[non_zero_mask]
        
        # 3. Create new Post-Padded tensor
        max_len, features = raw_tensor.shape
        new_tensor = np.zeros((max_len, features), dtype=np.float32)
        new_tensor[:actual_length] = valid_data # Fill valid data at the START
        
        return torch.from_numpy(new_tensor).float(), actual_length

    def __getitem__(self, idx):
        obs = self.data[idx]
        
        # Process each input to get Tensor (Post-Padded) and Length
        x_drug, len_drug = self._process_sequence(obs['drug'])
        x_lab, len_lab = self._process_sequence(obs['lab'])
        x_diag, len_diag = self._process_sequence(obs['diagnosis'])
        
        return {
            'drug': x_drug,
            'drug_len': len_drug,
            'lab': x_lab,
            'lab_len': len_lab,
            'diagnosis': x_diag,
            'diagnosis_len': len_diag,
            'static': torch.from_numpy(obs['static']).float(),
            'label': torch.tensor(obs['label']).float().unsqueeze(0)
        }

def train_model():
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EHRDataset('train.npy')
    val_dataset = EHRDataset('validate.npy')
    
    sample = train_dataset[0]
    print("-------- Sample 0 inspection")

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor Shape {value.shape}, Type: {value.dtype}")
        else: 
            print(f"{key}: {value}")
    print("\n--- Drug Tensor Preview (First 5 rows) ---")
    print(sample['drug'][:5])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EnsembleHFPredictor().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
        for batch in loop:
            x_drug = batch['drug'].to(DEVICE)
            l_drug = batch['drug_len'] # Lengths stay on CPU usually for packing logic, but model handles it
            
            x_lab = batch['lab'].to(DEVICE)
            l_lab = batch['lab_len']
            
            x_diag = batch['diagnosis'].to(DEVICE)
            l_diag = batch['diagnosis_len']
            
            x_static = batch['static'].to(DEVICE)
            y = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass now includes lengths
            y_pred = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_drug, l_drug = batch['drug'].to(DEVICE), batch['drug_len']
                x_lab, l_lab = batch['lab'].to(DEVICE), batch['lab_len']
                x_diag, l_diag = batch['diagnosis'].to(DEVICE), batch['diagnosis_len']
                x_static = batch['static'].to(DEVICE)
                y = batch['label'].to(DEVICE)
                
                y_pred = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

if __name__ == '__main__':
    train_model()