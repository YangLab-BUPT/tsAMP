import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from micmodel import TSAMPC
import numpy as np

import os
import pandas as pd
import torch
from tqdm import tqdm

class DataLoader:
    def __init__(self, positive_dir, target_excel, target_base_dir):
        self.positive_mean_representations = []
        self.miclabels = []  
        self.positive_name = []
        self.target_name = []
        self.target_mapping_df = self.load_target_mapping(target_excel)  
        self.load_positive_data(positive_dir, target_base_dir)

    def load_target_mapping(self, target_excel):
        df = pd.read_excel(target_excel, header=0)
        return df  

    def load_positive_data(self, positive_dir, target_base_dir):
        positive_files = os.listdir(positive_dir)
        
        for file in tqdm(positive_files, desc="Loading positive data", unit="file"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(positive_dir, file))
                label = file[:-3]  
                
                if label in self.target_mapping_df.iloc[:, 0].values:  
                    target_row = self.target_mapping_df[self.target_mapping_df.iloc[:, 0] == label]  
                    
                    if not target_row.empty:  
                        target = target_row.iloc[0, 1] 
                        
                     
                        target_words = target.split()
                        if len(target_words) >= 2:
                            target = " ".join(target_words[:2])
                        else:
                            target = target 
                        
                        if target in self.target_mapping_df.iloc[:, 1].values:  
                            mic_label = target_row.iloc[0, 4]  
                            self.load_target_data(label, target, data['mean_representations'], mic_label)

    def load_target_data(self, label, target_label, positive_mean_representations, mic_label):
        target_folder = '/tsAMP/data/tsAMP-C/species'
     
        target_label = target_label.replace(" ", "_")
        
        target_path = os.path.join(target_folder, target_label)
        
        if os.path.isdir(target_path):
            target_files = [f for f in os.listdir(target_path) if f.endswith('.pt')]
            
            for target_file in target_files:
                target_data = torch.load(os.path.join(target_path, target_file))
                
                for target_representation in target_data['mean_representations'].values():
                    for mean_representation in positive_mean_representations.values():
                        combined_vector = torch.cat((mean_representation, target_representation))
                        
                        self.positive_name.append(label)
                     
                        self.target_name.append(target_label)
                     
                        self.positive_mean_representations.append(combined_vector)
                       
                        self.miclabels.append(mic_label)
                    

    def get_data(self):
        return self.positive_mean_representations, self.miclabels, self.target_name, self.positive_name

def predict_and_save_results(model, dataloader, model_path, output_xlsx):
    positive_mean_representations, positive_names = dataloader.get_data()
    
    results = []
    

    target_taxids = find_target_taxids(model_path, dataloader.taxid_mapping)
    target_taxids1 = list(target_taxids)  
    
    for i, positive_name in enumerate(positive_names):
        for taxid in target_taxids1:
            target_folder = os.path.join('/tsAMP/data/tsAMP-C/species', str(taxid).strip())
            if os.path.isdir(target_folder):
                target_files = [f for f in os.listdir(target_folder) if f.endswith('.pt')]
                if target_files:
                    target_data = torch.load(os.path.join(target_folder, target_files[0]))
                    target_representation = target_data['mean_representations']
                    target_tensor = list(target_representation.values())[0] 
                   
                    combined_representation = torch.cat((positive_mean_representations[i], target_tensor)).unsqueeze(0)  
                   
                    model.eval()
                    with torch.no_grad():
                        output = model(combined_representation)
                        probabilities = torch.softmax(output, dim=1)
                        _, predictions = torch.max(output, 1)
                        
                    results.append({
                        'positive': positive_name,
                        'target': target_folder,
                        'taxid': taxid,
                        'predicted_mic': predictions.cpu().numpy()
                    })

    df_results = pd.DataFrame(results)
    df_results.to_excel(output_xlsx, index=False)  
    print(f"Results saved to: {output_xlsx}")

parser = argparse.ArgumentParser(description='Predict MIC labels and save results to Excel.')
parser.add_argument('--output_excel', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--test_dir', type=str)
args = parser.parse_args()

positive_dir = '/tsAMP/data/AMPesm1v'
target_base_dir = '/tsAMP/data/tsAMP-C/species'
target_excel = args.test_dir

data_loader = DataLoader(positive_dir, target_excel, target_base_dir)

model = TSAMPC()
model.load_state_dict(torch.load(args.model_path))  
model.eval()

predict_and_save_results(model, data_loader, args.output_excel)