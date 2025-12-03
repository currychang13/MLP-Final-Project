import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EHRDataset(Dataset):
    def __init__(self, processed_data_path):
        # Load the saved structured numpy array
        self.data = np.load(processed_data_path, allow_pickle=True)
        print(f"Loaded {len(self.data)} observations from {processed_data_path}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = self.data[idx]

        return {
            'drug': torch.from_numpy(obs['drug']).float(),
            'lab': torch.from_numpy(obs['lab']).float(),
            'diagnosis': torch.from_numpy(obs['diagnosis']).float(),
            'static': torch.from_numpy(obs['static']).float(),
            'label': torch.tensor(obs['label']).float().unsqueeze(0) # (1,) for BCELoss
        }