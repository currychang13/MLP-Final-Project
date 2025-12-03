import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

SEQUENCE_LENGTH = 100 

DRUG_DIM = 165     
LABS_DIM = 46      
DIAGNOSIS_DIM = 32 

STATIC_INPUT_DIM = 3 

# Hidden layer 
LSTM_HIDDEN_SIZE = 256
STATIC_MLP_OUTPUT = 16
FINAL_DENSE_OUTPUT = 16

class EnsembleHFPredictor(nn.Module):
    def __init__(self):
        super(EnsembleHFPredictor, self).__init__()
        
        # ------------ Single layer LSTM ---------------
        self.lstm_drug = nn.LSTM(input_size=DRUG_DIM, hidden_size=LSTM_HIDDEN_SIZE, batch_first=True)
        self.lstm_lab = nn.LSTM(input_size=LABS_DIM, hidden_size=LSTM_HIDDEN_SIZE, batch_first=True)
        self.lstm_diagnosis = nn.LSTM(input_size=DIAGNOSIS_DIM, hidden_size=LSTM_HIDDEN_SIZE, batch_first=True)
        
        # ---------------- MLP --------------------
        self.dense_static = nn.Sequential(
            nn.Linear(STATIC_INPUT_DIM, STATIC_MLP_OUTPUT),
            nn.ReLU()
        )
        
        # ensemble method
        ensemble_input_size = 3 * LSTM_HIDDEN_SIZE + STATIC_MLP_OUTPUT # 3*256 + 16 = 784
        
        # Final Dense Hidden Layer
        self.final_hidden = nn.Sequential(
            nn.Linear(ensemble_input_size, FINAL_DENSE_OUTPUT),
            nn.ReLU()
        )
        
        # Output Layer 
        self.classifier = nn.Sequential(
            nn.Linear(FINAL_DENSE_OUTPUT, 1),
            nn.Sigmoid() 
        )

    def forward(self, x_drug, x_lab, x_diagnosis, x_static):
        # Only use the hidden state of the last time step (h_n), original return tuple is (output, h_n, c_n)
        _, (h_drug, _) = self.lstm_drug(x_drug)
        drug_features = h_drug.squeeze(0)   # flatten to 2d, ignore the layer#
        
        _, (h_lab, _) = self.lstm_lab(x_lab)
        lab_features = h_lab.squeeze(0)
        
        _, (h_diagnosis, _) = self.lstm_diagnosis(x_diagnosis)
        diagnosis_features = h_diagnosis.squeeze(0)

        # Static        
        static_features = self.dense_static(x_static)
        
        # Concatenate all features
        combined_features = torch.cat(
            (drug_features, lab_features, diagnosis_features, static_features), 
            dim=1
        )
        
        # head
        hidden_output = self.final_hidden(combined_features)

        output = self.classifier(hidden_output)
        
        return output