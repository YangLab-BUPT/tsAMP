import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from model import  MLP
class DataLoader:
    def __init__(self, negative2_dir, excel2_file2,negative3_dir, excel3_file3):
        self.data = []
        self.labels = []
        self.physical_properties_negative2 = self.load_physical_properties(excel2_file2)
        self.load_samples_upper(negative2_dir, physical_properties=self.physical_properties_negative2) 
        self.physical_properties_negative3 = self.load_physical_properties(excel3_file3)
        self.load_samples_upper(negative3_dir, physical_properties=self.physical_properties_negative3) 
        
    def load_samples(self, data_dir, physical_properties):
        files = os.listdir(data_dir)
        for file in tqdm(files, desc=f"Loading negative samples"):
            if file.endswith('.pt'):
                
                data = torch.load(os.path.join(data_dir, file))
                mean_representation = torch.stack(list(data['mean_representations'].values()))
                
              
                mean_representation_flat = mean_representation.flatten()
                

                sample_label = file[:-3].upper() # testAPD3不加
                if sample_label in physical_properties.index:
                    
                    props = physical_properties.loc[sample_label].values[1:5]
                    props = torch.tensor(props.astype(float), dtype=torch.float32)
                    
                    combined_representation = torch.cat((mean_representation_flat, props), dim=0)
                  
                    self.data.append(combined_representation)
           
                    self.labels.append(sample_label) 
    def load_samples_upper(self, data_dir, physical_properties):
        files = os.listdir(data_dir)
        for file in tqdm(files, desc=f"Loading negative samples"):
            if file.endswith('.pt'):
                
                data = torch.load(os.path.join(data_dir, file))
                mean_representation = torch.stack(list(data['mean_representations'].values()))
                
              
                mean_representation_flat = mean_representation.flatten()
                

                sample_label = file[:-3].upper() 
                if sample_label in physical_properties.index:
                    
                    props = physical_properties.loc[sample_label].values[1:5]
                    props = torch.tensor(props.astype(float), dtype=torch.float32)
                    
                    combined_representation = torch.cat((mean_representation_flat, props), dim=0)
                    self.data.append(combined_representation)
                   
                    self.labels.append(sample_label) 
    def load_physical_properties(self, excel_file):
        df = pd.read_excel(excel_file)
        return df.set_index(df.columns[0]) 

    def get_data(self):
        if not self.data:
            print("Error: No samples found.")
            return None
        X = torch.stack(self.data)
        return X


def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = MLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model



model_path = 'tsAMP/model/tsAMP-I/tsAMP-I.pt'

test_dir = '/tsAMP/data/tsAMP-I/'
test_file = '/tsAMP/data/tsAMP-I/testLAMP.xlsx'

data_loader = DataLoader(test_dir, test_file)
X_data = data_loader.get_data()

input_dim =1284  
hidden_dim = 256
output_dim = 1

model = load_model(model_path, input_dim, hidden_dim, output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if X_data is not None:
    X_data = X_data.to(device) 
    with torch.no_grad():
        predictions = model(X_data)
        predicted_labels = torch.sigmoid(predictions).cpu().numpy()  

        df = pd.DataFrame({
            'Sample_Label': data_loader.labels, 
            'Predicted': predicted_labels.flatten()
        })
        output_csv_path = 'positive_predictions.csv'
        df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")