import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# Feature dimension
DRUG_DIM = 198     
LABS_DIM = 47      
DIAGNOSIS_DIM = 32 
STATIC_DIM = 3 

# STATIC
MLP_HIDDEN_1 = 256          # 128
MLP_HIDDEN_2 = 128
MLP_OUTPUT = 16

# LSTMs
LSTM_HIDDEN_SIZE = 512
LSTM_LAYER = 1              # 10

# Final Dense layer
FINAL_DENSE_HIDDEN_1 = 256
FINAL_DENSE_HIDDEN_2 = 128
FINAL_DENSE_OUTPUT = 32     # 64

class EnsembleHFPredictor(nn.Module):
    def __init__(self):
        super(EnsembleHFPredictor, self).__init__()
        
        # ------------ LSTM ---------------
        self.lstm_drug = nn.LSTM(
            input_size=DRUG_DIM, 
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYER, 
            batch_first=True, 
            # bidirectional=True, 
            dropout=0)
        
        self.lstm_lab = nn.LSTM(
            input_size=LABS_DIM, 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=LSTM_LAYER, 
            batch_first=True, 
            # bidirectional=True, 
            dropout=0)
        
        self.lstm_diagnosis = nn.LSTM(
            input_size=DIAGNOSIS_DIM, 
            hidden_size=LSTM_HIDDEN_SIZE, 
            num_layers=LSTM_LAYER, 
            batch_first=True, 
            # bidirectional=True,
            dropout=0)
        
        # ---------------- MLP --------------------
        self.dense_static = nn.Sequential(
            nn.Linear(STATIC_DIM, MLP_HIDDEN_1),
            nn.ReLU(),   
            
            nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2),
            nn.ReLU(),
            
            nn.Dropout(0.3),
            nn.Linear(MLP_HIDDEN_2, MLP_OUTPUT),
            nn.ReLU(),
        )
        
        ensemble_input_size =  3 * LSTM_HIDDEN_SIZE + MLP_OUTPUT
        
        # Final Dense Hidden Layer
        self.final_hidden = nn.Sequential(
            nn.BatchNorm1d(ensemble_input_size),

            nn.Linear(ensemble_input_size, FINAL_DENSE_HIDDEN_1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(FINAL_DENSE_HIDDEN_1, FINAL_DENSE_HIDDEN_2),
            # nn.BatchNorm1d(FINAL_DENSE_HIDDEN_2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(FINAL_DENSE_HIDDEN_2, FINAL_DENSE_OUTPUT)
        )
        
        # Output Layer 
        self.classifier = nn.Linear(FINAL_DENSE_OUTPUT, 2)

    def _process_packed_input(self, lstm_layer, x_tensor, lengths):
        packed_input = rnn_utils.pack_padded_sequence(
            x_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = lstm_layer(packed_input)  # implicit state, Initialize h_0, c_0 at each batch.
        
        # return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]
    
    def forward(self, x_drug, drug_lens, x_lab, lab_lens, x_diagnosis, diag_lens, x_static):
        
        drug_features = self._process_packed_input(self.lstm_drug, x_drug, drug_lens)
        lab_features = self._process_packed_input(self.lstm_lab, x_lab, lab_lens)
        diagnosis_features = self._process_packed_input(self.lstm_diagnosis, x_diagnosis, diag_lens)
        
        static_features = self.dense_static(x_static)
        
        combined_features = torch.cat(
            (drug_features, lab_features, diagnosis_features, static_features), 
            dim=1
        )
        
        hidden_output = self.final_hidden(combined_features)
        output = self.classifier(hidden_output)
        
        return output