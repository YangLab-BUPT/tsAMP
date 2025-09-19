import torch
import torch.nn as nn 
import torch.optim as optim
from dataloader1 import DataLoader
from model import  MLP
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score
import pandas as pd
import random

            
            
def calculate_metrics(outputs, targets):
    probs = torch.sigmoid(outputs).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    targets = targets.cpu().numpy()
    precision = precision_score(targets, preds)
    accuracy = accuracy_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return precision,accuracy, recall, f1

def train_model(X_train, y_train, X_val, y_val, model, device, num_epochs=100, learning_rate=0.00005, batch_size=32, results_file='useesm2/gmgc_training_results2.csv'):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    num_train_batches = len(X_train) // batch_size
    num_val_batches = len(X_val) // batch_size

    training_losses = []
    validation_losses = []
    accuracies = []
    precisions =[]
    recalls = []
    f1_scores = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        model.train()
        train_loss = 0
        train_loader = tqdm(range(num_train_batches), desc="Training", leave=False)
        for i in train_loader:
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train[start_idx:end_idx].to(device)  # Move to GPU
            y_batch = y_train[start_idx:end_idx].to(device).unsqueeze(1)  # Move to GPU
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loader.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        all_val_outputs = []
        all_val_targets = []

        with torch.no_grad():
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_val_batch = X_val[start_idx:end_idx].to(device)  # Move to GPU
                y_val_batch = y_val[start_idx:end_idx].to(device).unsqueeze(1)  # Move to GPU

                val_outputs = model(X_val_batch)
                loss = criterion(val_outputs, y_val_batch)
                val_loss += loss.item()

                all_val_outputs.append(val_outputs)
                all_val_targets.append(y_val_batch)

        all_val_outputs = torch.cat(all_val_outputs).cpu()
        all_val_targets = torch.cat(all_val_targets).cpu()

        precision, accuracy, recall, f1 = calculate_metrics(all_val_outputs, all_val_targets)

        training_losses.append(train_loss / num_train_batches)
        validation_losses.append(val_loss / num_val_batches)
        precisions.append(precision)
        accuracies.append(accuracy)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_losses[-1]:.4f}, '
              f'Validation Loss: {validation_losses[-1]:.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        torch.save(model.state_dict(), f'useesm2/epoch/{epoch + 1}.pt')

    results_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'Training Loss': training_losses,
        'Validation Loss': validation_losses,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })
    results_df.to_csv(results_file, index=False)

positive_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/trainpositive_filtered'  
excel_file = '/data2/lhmData/AMP/dataprocess/charge/positive.xlsx'  
negative_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/negative_train'
excel_file2 = '/data2/lhmData/AMP/dataprocess/charge/negative_review_5_train.xlsx' 
negative1_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/negative_train'
excel1_file2 = '/data2/lhmData/AMP/dataprocess/charge/negative_review_c_train.xlsx'
negative2_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/negative_train'
excel2_file2 = '/data2/lhmData/AMP/dataprocess/charge/negative_gmgc_5_train.xlsx'
negative3_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/negative_train'
excel3_file2 = '/data2/lhmData/AMP/dataprocess/charge/negative_gmgc_c_train.xlsx'''

data_loader = DataLoader(positive_dir, negative_dir, negative3_dir,negative1_dir, negative2_dir,excel_file, excel_file2, excel3_file2,excel1_file2,excel2_file2)

X_train, X_val, y_train, y_val = data_loader.get_data()

input_dim = 1284  
output_dim = 1  

model = MLP(input_dim, 256, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) 
X_train = X_train.to(device)  
y_train = y_train.to(device)  
X_val = X_val.to(device)  
y_val = y_val.to(device)  
train_model(X_train, y_train, X_val, y_val, model, device)