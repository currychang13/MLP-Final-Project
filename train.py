import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from model import EnsembleHFPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import os

BATCH_SIZE = 32
LEARNING_RATE = 3e-5    #2e-5
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "./plots"
WEIGHT_DIR = "./weights"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# dynamically pads to LONGEST sequence in the batch.
def ehr_collate_fn(batch):
    """
    Args:
        batch: A list of dictionaries (results of __getitem__)
               [
                 {'drug': Tensor(5, 165), 'label': ...}, 
                 {'drug': Tensor(10, 165), 'label': ...}, 
                 ...
               ]
    Returns:
        A dictionary containing the batched (padded) tensors.
    """
    batch_out = {}
    
    for key in ['drug', 'lab', 'diagnosis']:
        tensors = [item[key] for item in batch]
        padded_batch = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        
        batch_out[key] = padded_batch
        
        len_key = f"{key}_len" 
        batch_out[len_key] = torch.tensor([item[len_key] for item in batch])

    # static
    batch_out['static'] = torch.stack([item['static'] for item in batch])
    batch_out['label'] = torch.stack([item['label'] for item in batch])
    
    return batch_out

class EHRDataset(Dataset):
    def __init__(self, processed_data_path):
        self.data = np.load(processed_data_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = self.data[idx]
        return {
            'drug': torch.from_numpy(obs['drug']).float(),
            'drug_len': obs['drug_len'], 
            'lab': torch.from_numpy(obs['lab']).float(),
            'lab_len': obs['lab_len'],
            'diagnosis': torch.from_numpy(obs['diagnosis']).float(),
            'diagnosis_len': obs['diag_len'], 
            'static': torch.from_numpy(obs['static']).float(),
            'label': torch.tensor(obs['label']).float() 
        }

def plot_history(history):

    os.makedirs(PLOT_DIR, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
# --- Plot 1: Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "loss.png"))
    plt.close() # Close to free memory and start fresh canvas

    # --- Plot 2: ROC-AUC ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_roc_auc'], label='Val ROC-AUC', color='orange')
    plt.title('Validation ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "roc_auc.png"))
    plt.close()

    # --- Plot 3: PR-AUC ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_pr_auc'], label='Val PR-AUC', color='green')
    plt.title('Validation PR-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "pr_auc.png"))
    plt.close()

    print(f"Plots saved to {PLOT_DIR}/ directory (loss.png, roc_auc.png, pr_auc.png)")

def train_model():
    train_dataset = EHRDataset('train.npy')
    val_dataset = EHRDataset('validate.npy')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ehr_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ehr_collate_fn)

    model = EnsembleHFPredictor().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-9)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_roc_auc': [],
        'val_pr_auc': []
    }

    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        
        for batch in loop:
            x_drug = batch['drug'].to(DEVICE)
            x_diag = batch['diagnosis'].to(DEVICE)
            x_lab = batch['lab'].to(DEVICE)
            x_static = batch['static'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            
            l_drug, l_lab, l_diag = batch['drug_len'], batch['lab_len'], batch['diagnosis_len']
            
            optimizer.zero_grad()
            logits = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
            
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
            optimizer.step()
        
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        
        all_y_true = []
        all_y_scores = []

        with torch.no_grad():
            for batch in val_loader:
                x_drug = batch['drug'].to(DEVICE)
                x_lab = batch['lab'].to(DEVICE)
                x_diag = batch['diagnosis'].to(DEVICE)
                x_static = batch['static'].to(DEVICE)
                y = batch['label'].to(DEVICE)
                
                l_drug, l_lab, l_diag = batch['drug_len'], batch['lab_len'], batch['diagnosis_len']
                
                logits = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
                
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                
                all_y_true.append(y.cpu().numpy())
                all_y_scores.append(probs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        all_y_true = np.vstack(all_y_true)
        all_y_scores = np.vstack(all_y_scores)
        val_preds = (all_y_scores > 0.5).astype(int)
        val_acc = accuracy_score(all_y_true, val_preds)

        epoch_roc_auc = roc_auc_score(all_y_true, all_y_scores, average='macro')
        epoch_pr_auc = average_precision_score(all_y_true, all_y_scores, average='macro')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_roc_auc'].append(epoch_roc_auc)
        history['val_pr_auc'].append(epoch_pr_auc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            save_path = os.path.join(WEIGHT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved (Val loss: {best_val_loss:.4f})")
        # else:
        #     counter +=1
        #     print(f"--> No improvement. EarlyStopping counter: {counter} out of {patience}")        
        #     if counter >= patience:
        #         print("=============================================")
        #         print("Early stopping triggered. Training terminated.")
        #         print("=============================================")
        #         break 

            
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model.pth"))
        print("===============================================================================================")
   
    plot_history(history)
    print(f"Training complete. Weights saved in {WEIGHT_DIR}/")

if __name__ == '__main__':
    train_model()