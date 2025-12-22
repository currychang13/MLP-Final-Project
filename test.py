import torch
from torch.utils.data import DataLoader
from train import EHRDataset, ehr_collate_fn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, accuracy_score
from model import EnsembleHFPredictor
from tqdm import tqdm
import os

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA_PATH = "test.npy"
PLOT_DIR = "./result"

def plot_confusion_matrix(cm, class_name, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f"Confusion Matrix: {class_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(chkpt_path):
    MODEL = os.path.basename(chkpt_path).strip(".pth")
    
    print(f"Testing on device: {DEVICE}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    test_dataset = EHRDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ehr_collate_fn)
    
    model = EnsembleHFPredictor().to(DEVICE)

    if not os.path.exists(chkpt_path):
        print(f"Error: {chkpt_path} not found.")
        return
        
    print(f"Loading weights from {chkpt_path}...")
    model.load_state_dict(torch.load(chkpt_path, map_location=DEVICE))
    model.eval()

    all_y_true = []
    all_y_scores = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x_drug = batch['drug'].to(DEVICE)
            x_lab = batch['lab'].to(DEVICE)
            x_diag = batch['diagnosis'].to(DEVICE)
            x_static = batch['static'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            
            l_drug, l_lab, l_diag = batch['drug_len'], batch['lab_len'], batch['diagnosis_len']

            logits = model(x_drug, l_drug, x_lab, l_lab, x_diag, l_diag, x_static)
            probs = torch.sigmoid(logits)
            # probs[:, 1] = torch.max(probs[:, 0], probs[:,1])
            
            all_y_true.append(y.cpu().numpy())
            all_y_scores.append(probs.cpu().numpy())

    y_true = np.vstack(all_y_true)
    y_scores = np.vstack(all_y_scores)
    y_pred_binary = (y_scores > 0.5).astype(int)
    
    target_names = ['Death_30_Days', 'Death_180_Days']
    
    roc_auc = roc_auc_score(y_true, y_scores, average='macro')
    pr_auc = average_precision_score(y_true, y_scores, average='macro')
    acc = accuracy_score(y_true, y_pred_binary) 
    
    report_str = classification_report(y_true, y_pred_binary, target_names=target_names, zero_division=0)

    output = []
    output.append("="*40)
    output.append("       TEST SET PERFORMANCE       ")
    output.append("="*40)
    output.append(f"Overall Accuracy: {acc:.4f} (Exact Match)")
    output.append(f"Overall ROC-AUC:  {roc_auc:.4f}")
    output.append(f"Overall PR-AUC:   {pr_auc:.4f}")
    output.append("-" * 40)
    output.append("Classification Report:\n")
    output.append(report_str)
    output.append("-" * 40)
    output.append("CONFUSION MATRICES SUMMARY:\n")

    for i, name in enumerate(target_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        
        output.append(f"--- {name} (AUC: {auc:.4f}) ---")
        output.append(f"TP: {cm[1][1]} | FP: {cm[0][1]}")
        output.append(f"FN: {cm[1][0]} | TN: {cm[0][0]}\n")
        
        plot_confusion_matrix(cm, name, f"{MODEL}_cm_{name}.png")

    final_output = "\n".join(output)
    print(final_output)
    FILENAME = MODEL + "_report.txt"
    REPORT_FILE = os.path.join(PLOT_DIR, FILENAME)
    with open(REPORT_FILE, "w") as f:
        f.write(final_output)
    
    print(f"\nReport saved to {REPORT_FILE}")
    print(f"Confusion matrices saved to {PLOT_DIR}")

if __name__ == '__main__':
    chkpt_path = ["./weights/best_model.pth", "./weights/early_stopping.pth"]
    for chkpt in chkpt_path:
        evaluate_model(chkpt)