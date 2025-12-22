import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import EHRDataset, ehr_collate_fn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from model import EnsembleHFPredictor
from tqdm import tqdm
import os
import shap
import pandas as pd

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA_PATH = "test.npy"
PLOT_DIR = "./result/ablation"

def plot_bar(data_dict, title, filename):
    plt.figure(figsize=(8, 6))
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    sns.barplot(x=keys, y=values, palette='viridis', hue=keys, legend=False)
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y') 
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

class ShapWrapper(nn.Module):
    def __init__(self, model, l_drug, l_lab, l_diag, target_idx: int):
        super().__init__()
        self.model = model
        self.l_drug_full = l_drug.cpu()
        self.l_lab_full  = l_lab.cpu()
        self.l_diag_full = l_diag.cpu()
        self.target_idx = target_idx  # 0 or 1

    def forward(self, x_drug, x_lab, x_diag, x_static):
        n = x_drug.size(0)

        self.model.lstm_drug.train()
        self.model.lstm_lab.train()
        self.model.lstm_diagnosis.train()

        l_drug = self.l_drug_full[:n]
        l_lab  = self.l_lab_full[:n]
        l_diag = self.l_diag_full[:n]

        l_drug = torch.clamp(l_drug, min=1, max=x_drug.size(1))
        l_lab  = torch.clamp(l_lab,  min=1, max=x_lab.size(1))
        l_diag = torch.clamp(l_diag, min=1, max=x_diag.size(1))

        out = self.model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)  # [B,2]
        return out[:, self.target_idx:self.target_idx+1]  # [B,1]
 

def set_rnn_train_only(model):
    model.eval() 
    model.lstm_drug.train()
    model.lstm_lab.train()
    model.lstm_diagnosis.train()
    
def ensure_btd(lab_shap):
    lab_shap = np.array(lab_shap)

    if lab_shap.ndim == 4:
        if lab_shap.shape[-1] == 1:
            lab_shap = lab_shap[..., 0]     
        elif lab_shap.shape[0] == 1:
            lab_shap = lab_shap[0]           
        else:
            raise ValueError(f"Unexpected 4D lab_shap shape: {lab_shap.shape}")

    if lab_shap.ndim != 3:
        raise ValueError(f"Unexpected lab_shap ndim={lab_shap.ndim}, shape={lab_shap.shape}")

    return lab_shap

def run_shap_analysis(model, dataloader, device, lab_feature_names, nsamples=10):
    set_rnn_train_only(model)

    all_lab_shap_agg = {0: [], 1: []}  
    all_lab_x_agg    = {0: [], 1: []}   
    totals = {0: None, 1: None}
    counts = {0: 0, 1: 0}

    for batch in tqdm(dataloader, desc="SHAP over test"):
        x_drug = batch['drug'].to(device)
        x_lab  = batch['lab'].to(device)
        x_diag = batch['diagnosis'].to(device)
        x_stat = batch['static'].to(device)

        l_drug = batch['drug_len']
        l_lab  = batch['lab_len']
        l_diag = batch['diagnosis_len']

        k = min(8, x_lab.size(0))
        background = [x_drug[:k], x_lab[:k], x_diag[:k], x_stat[:k]]

        inputs = [x_drug, x_lab, x_diag, x_stat]

        T = x_lab.size(1)
        mask = (torch.arange(T, device=device)[None, :] < l_lab.to(device)[:, None]).float().cpu().numpy()[..., None]

        for target_idx in (0, 1):
            set_rnn_train_only(model)

            shap_model = ShapWrapper(model, l_drug, l_lab, l_diag, target_idx=target_idx).to(device)

            explainer = shap.GradientExplainer(shap_model, background)
            shap_vals = explainer.shap_values(inputs, nsamples=nsamples)

            lab_shap = ensure_btd(shap_vals[1])  # -> [B,T,D]
            B, T2, D = lab_shap.shape

            if T2 != mask.shape[1]:
                mask2 = mask[:, :T2, :]
            else:
                mask2 = mask

            abs_or_signed = lab_shap  
            valid_steps = mask2.sum(axis=1)            
            lab_shap_agg = (abs_or_signed * mask2).sum(axis=1) / np.clip(valid_steps, 1.0, None) 

            lab_x = x_lab.detach().cpu().numpy()          
            lab_x_agg = (lab_x[:, :T2, :] * mask2).sum(axis=1) / np.clip(valid_steps, 1.0, None)

            all_lab_shap_agg[target_idx].append(lab_shap_agg)
            all_lab_x_agg[target_idx].append(lab_x_agg)

            per_sample_abs = np.abs(lab_shap_agg)           
            batch_mean = per_sample_abs.mean(axis=0)       

            if totals[target_idx] is None:
                totals[target_idx] = batch_mean
            else:
                totals[target_idx] += batch_mean
            counts[target_idx] += 1

    for target_idx in (0, 1):
        feature_importance = totals[target_idx] / max(counts[target_idx], 1)

        indices = np.argsort(feature_importance)[-20:]
        feat_names = [lab_feature_names[i] for i in indices]

        target = "Death_30" if target_idx == 0 else "Death_180"

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), feat_names)
        plt.xlabel("mean_t(mean_batch(|SHAP|))")
        plt.title(f"Top 20 Important Lab Features — label {target}")
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"shap_lab_importance_label_{target}.png"))
        plt.close()

        shap_mat = np.concatenate(all_lab_shap_agg[target_idx], axis=0)
        x_mat    = np.concatenate(all_lab_x_agg[target_idx], axis=0)

        plt.figure()
        shap.summary_plot(
            shap_mat,
            features=x_mat,
            feature_names=lab_feature_names,
            show=False,
            plot_type="dot",
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"shap_summary_beeswarm_label_{target}.png"), dpi=200)
        plt.close()

def run_ablation_test(model, dataloader, device):
    print("\n" + "="*40)
    print("       STARTING ABLATION STUDY       ")
    print("="*40)
    
    ablation_modes = ['All_Features', 'No_Drug', 'No_Lab', 'No_Diagnosis', 'No_Static']
    results = {}

    for mode in ablation_modes:
        print(f"Testing Mode: {mode}...")
        
        all_y_true = []
        all_y_scores = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {mode}", leave=False):
                x_drug = batch['drug'].to(device)
                x_lab = batch['lab'].to(device)
                x_diag = batch['diagnosis'].to(device)
                x_static = batch['static'].to(device)
                y = batch['label'].to(device)
                
                l_drug, l_lab, l_diag = batch['drug_len'], batch['lab_len'], batch['diagnosis_len']

                if mode == 'No_Drug':
                    x_drug = torch.zeros_like(x_drug)
                elif mode == 'No_Lab':
                    x_lab = torch.zeros_like(x_lab)
                elif mode == 'No_Diagnosis':
                    x_diag = torch.zeros_like(x_diag)
                elif mode == 'No_Static':
                    x_static = torch.zeros_like(x_static)

                logits = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
                probs = torch.sigmoid(logits)
                
                all_y_true.append(y.cpu().numpy())
                all_y_scores.append(probs.cpu().numpy())

        y_true = np.vstack(all_y_true)
        y_scores = np.vstack(all_y_scores)
        
        roc_auc = roc_auc_score(y_true, y_scores, average='macro')
        mode = mode.replace("_", " ")
        results[mode] = roc_auc
        print(f" -> Result: ROC-AUC = {roc_auc:.4f}")

    # Plot results
    plot_bar(results, "Ablation Study on Ensemble Components", "ablation_impact.png")
    
    return results

def main(chkpt_path):
    print(f"Testing on device: {DEVICE}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    test_dataset = EHRDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ehr_collate_fn)
    
    df_lab = pd.read_csv("./data/final_lab.csv")
    lab_cols = [c for c in df_lab.columns if c not in ['PERSONID2', 'LOGDATE']]
    
    model = EnsembleHFPredictor().to(DEVICE)
    if not os.path.exists(chkpt_path):
        print(f"Error: {chkpt_path} not found.")
        return
        
    print(f"Loading weights from {chkpt_path}...")
    model.load_state_dict(torch.load(chkpt_path, map_location=DEVICE))


    run_ablation_test(model, test_loader, DEVICE)
    run_shap_analysis(model, test_loader, DEVICE, lab_feature_names=lab_cols)

if __name__ == '__main__':
    chkpt = "./weights/best_model.pth"
    main(chkpt)