import torch
import torch.nn as nn
from dataset import EHRDataset
import numpy as np
from model import EnsembleHFPredictor 

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("Starting model training...")
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0
        for batch_data in train_loader:
            # Move data to device
            x_drug = batch_data['drug'].to(device)
            x_lab = batch_data['lab'].to(device)
            x_diagnosis = batch_data['diagnosis'].to(device)
            x_static = batch_data['static'].to(device)
            y_target = batch_data['label'].to(device)

            optimizer.zero_grad()
            
            y_pred = model(x_drug, x_lab, x_diagnosis, x_static)
            loss = criterion(y_pred, y_target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                x_drug = batch_data['drug'].to(device)
                x_lab = batch_data['lab'].to(device)
                x_diagnosis = batch_data['diagnosis'].to(device)
                x_static = batch_data['static'].to(device)
                y_target = batch_data['label'].to(device)
                
                y_pred = model(x_drug, x_lab, x_diagnosis, x_static)
                val_loss += criterion(y_pred, y_target).item()

        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    full_dataset = EHRDataset('final_processed_ehr_tensors.npy')
    
    # Split the dataset (80% Train, 10% Validation, 10% Test - mimicking the paper)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the model
    model = EnsembleHFPredictor()
    
    # Loss Function (Binary Cross-Entropy Loss)
    criterion = nn.BCELoss() 
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

    print("\nTraining completed. The test_loader is ready for final evaluation metrics (AUC/C-statistic).")