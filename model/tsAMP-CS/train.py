import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dataloadertrain import DataLoader as CustomDataLoader
from model import TSAMPCS
from tqdm import tqdm  
import pandas as pd 
import argparse
import os
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Predict MIC labels and save results to Excel.')
parser.add_argument('--train_dir', type=str)
parser.add_argument('--filename_dir', type=str)
args = parser.parse_args()
target_excel = args.train_dir
filename1 = args.filename_dir

def load_data():
    positive_dir = '/tsAMP/data/target_strains/positive'
    target_base_dir = '/tsAMP/data/target_strains/'
    data_loader = CustomDataLoader(positive_dir, target_excel, target_base_dir)
    positive_samples = data_loader.positive_mean_representations
    positive_labels = data_loader.miclabels 
    positive_labels = torch.tensor(list(positive_labels), dtype=torch.float32).to(device)  
    return positive_samples, positive_labels

def validate_model(model, X_val, y_val):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        outputs = model(X_val)
        val_preds = outputs.cpu().numpy()
        y_val_cpu = y_val.cpu().numpy()
        
        loss = criterion_reg(outputs, y_val.view(-1, 1).to(device))  
        total_loss += loss.item()

        mse = mean_squared_error(y_val_cpu, val_preds)
        r2 = r2_score(y_val_cpu, val_preds)

    return mse, r2

import torch.optim as optim
from torch.optim import lr_scheduler
save_dir = f'../mic_regression/717tsamp/epoch/{filename1}'

if os.path.exists(save_dir):
    print(f"Directory {save_dir} already exists. Skipping the entire training process.")
else:
    positive_data, positive_labels = load_data()
    all_data = torch.stack(positive_data).to(device)  
    all_labels = positive_labels

    X_train, X_test, y_train, y_test = train_test_split(all_data.cpu(), all_labels.cpu(), test_size=0.2, random_state=42, shuffle=True)
    train_dataset = TensorDataset(X_train.to(device), y_train.view(-1, 1).to(device)) 
    val_dataset = TensorDataset(X_test.to(device), y_test.view(-1, 1).to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

    model = TSAMPCS.to(device) 
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    results = []  
    num_epoch = 70

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch}', unit='batch'):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion_reg(outputs, labels.view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_loss += loss.item()

        total_val_loss = 0
        for val_inputs, val_labels in val_loader:
            val_loss = criterion_reg(model(val_inputs.to(device)), val_labels.view(-1, 1).to(device))
            total_val_loss += val_loss.item()

        train_mse, train_r2 = validate_model(model, X_train.to(device), y_train.to(device))
        val_mse, val_r2 = validate_model(model, X_test.to(device), y_test.to(device))

        print('lr: %.4f, epoch: %d'%(scheduler.get_lr()[0], epoch))
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epoch}], '
              f'Training Loss: {epoch_loss / len(train_loader):.4f}, '
              f'Validation Loss: {total_val_loss / len(val_loader):.4f}, '
              f'MSE (Training): {train_mse:.4f}, R2 Score (Training): {train_r2:.4f}, '
              f'MSE (Validation): {val_mse:.4f}, R2 Score (Validation): {val_r2:.4f}')

        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{save_dir}/mic_{epoch + 1}.pt')

        results.append({
            'Epoch': epoch + 1,
            'Training Loss': epoch_loss / len(train_loader),
            'Validation loss': total_val_loss / len(val_loader),
            'MSE (Training)': train_mse,
            'R2 Score (Training)': train_r2,
            'MSE (Validation)': val_mse,
            'R2 Score (Validation)': val_r2
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'../mic_regression/717tsamp/training/mic_training_{filename1}.csv', index=False)
    print("Training results saved to 'mic_training.csv'.")