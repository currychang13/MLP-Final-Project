import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# SEQUENCE_LENGTH = 100 

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
        self.classifier = nn.Linear(FINAL_DENSE_OUTPUT, 2)

    def _process_packed_input(self, lstm_layer, x_tensor, lengths):
        # Pack the sequence
        packed_input = rnn_utils.pack_padded_sequence(
            x_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = lstm_layer(packed_input)  # implicit state, Initialize h_0, c_0 at each batch.
        
        return h_n.squeeze(0)
    
    def forward(self, x_drug, drug_lens, x_lab, lab_lens, x_diagnosis, diag_lens, x_static):
        
        # 1. Process LSTMs with Packing
        drug_features = self._process_packed_input(self.lstm_drug, x_drug, drug_lens)
        lab_features = self._process_packed_input(self.lstm_lab, x_lab, lab_lens)
        diagnosis_features = self._process_packed_input(self.lstm_diagnosis, x_diagnosis, diag_lens)
        
        # 2. Process Static Data
        static_features = self.dense_static(x_static)
        
        # 3. Feature Fusion
        combined_features = torch.cat(
            (drug_features, lab_features, diagnosis_features, static_features), 
            dim=1
        )
        
        # 4. Classification
        hidden_output = self.final_hidden(combined_features)
        output = self.classifier(hidden_output)
        
        return output