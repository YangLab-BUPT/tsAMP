import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataloadermic import DataLoader as CustomDataLoader
from micmodel import MIC_Transformer
from tqdm import tqdm  
import pandas as pd 
import argparse
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.tanh(x)  

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x) 

parser = argparse.ArgumentParser(description='Predict MIC labels and save results to Excel.')
parser.add_argument('--train_dir', type=str)
parser.add_argument('--filename_dir', type=str)
parser.add_argument('--mic', type=str)
args = parser.parse_args()
target_excel = args.train_dir
filename1 = args.filename_dir
mic=args.mic

def load_data():
    positive_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/positive'
    target_base_dir = '/data2/lhmData/AMP/dataprocess/ESM1v/ncbitarget'
    target_base_dir2='/data2/lhmData/AMP/dataprocess/ESM1v/ncbitarget_new'
  
    data_loader = CustomDataLoader(positive_dir, target_excel, target_base_dir,target_base_dir2)
    
    samples = data_loader.positive_mean_representations 
    labels = data_loader.miclabels 

    all_data = torch.stack(samples).to(device)
    all_labels = torch.tensor(labels, dtype=torch.long).to(device) 

    return all_data, all_labels

def validate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        loss = criterion(outputs, y_val)
        _, val_preds = torch.max(outputs, 1)
        y_val_cpu = y_val.cpu().numpy()
        val_preds_cpu = val_preds.cpu().numpy()
        precision = precision_score(y_val_cpu, val_preds_cpu, average='binary', zero_division=1)
        recall = recall_score(y_val_cpu, val_preds_cpu, average='binary', zero_division=1)
        f1 = f1_score(y_val_cpu, val_preds_cpu, average='binary', zero_division=1)
        accuracy = accuracy_score(y_val_cpu, val_preds_cpu)
    return accuracy, precision, recall, f1, loss.item()
save_dir = f'/data2/lhmData/AMP/train/chem/result/finalepoch/GAN1/{filename1}'
if os.path.exists(save_dir):
    print(f"Directory {save_dir} already exists. Skipping the entire training process.")
else:
    os.makedirs(save_dir, exist_ok=True)

    all_data, all_labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(all_data.cpu(), all_labels.cpu(), test_size=0.2, random_state=32, stratify=all_labels.cpu())
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device)) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MIC_Transformer().to(device) 
    generator = Generator(latent_dim=100, output_dim=X_train.shape[1]).to(device)
    discriminator = Discriminator(input_dim=X_train.shape[1]).to(device)

    criterion = torch.nn.CrossEntropyLoss() 
    criterion_gan = torch.nn.BCELoss() 
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results = []  
    num_epoch = 50

    for epoch in range(num_epoch): 
     model.train()
     generator.train()
     discriminator.train()
     epoch_loss = 0
     epoch_classification_loss = 0

     for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch}', unit='batch'):
        current_batch_size = inputs.size(0)
        optimizer_D.zero_grad()
        real_data = inputs.to(device)
        real_labels_gan = torch.ones(current_batch_size, 1).to(device)
        z = torch.randn(current_batch_size, 100).to(device)
        fake_data = generator(z)
        fake_labels_gan = torch.zeros(current_batch_size, 1).to(device)
        real_output = discriminator(real_data)
        d_loss_real = criterion_gan(real_output, real_labels_gan)
        
        fake_output = discriminator(fake_data.detach())
        d_loss_fake = criterion_gan(fake_output, fake_labels_gan)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = criterion_gan(fake_output, real_labels_gan)  
        g_loss.backward()
        optimizer_G.step()
        optimizer.zero_grad()
        synthetic_labels = torch.ones(current_batch_size, dtype=torch.long).to(device)
        combined_inputs = torch.cat([inputs, fake_data.detach()], dim=0)
        combined_labels = torch.cat([labels, synthetic_labels], dim=0)
        shuffle_idx = torch.randperm(combined_inputs.size(0))
        combined_inputs = combined_inputs[shuffle_idx]
        combined_labels = combined_labels[shuffle_idx]
        outputs = model(combined_inputs)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()
        
        epoch_classification_loss += loss.item()
        epoch_loss += d_loss.item() + g_loss.item()

     accuracy, precision, recall, f1, val_loss = validate_model(model, X_test.to(device), y_test.to(device))
     print(f'Epoch [{epoch + 1}/{num_epoch}], Training Loss: {epoch_loss / len(train_loader):.4f}, '
          f'Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

    
     torch.save(model.state_dict(), f'{save_dir}/mic{mic}_{epoch + 1}.pt')
     results.append({
        'Epoch': epoch + 1,
        'Training Loss': epoch_loss / len(train_loader),
        'Validation Loss': val_loss,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
     })

    results_df = pd.DataFrame(results)
    results_df.to_csv('mic_training.csv', index=False)
    print("Training results saved to 'mic_training.csv'.")