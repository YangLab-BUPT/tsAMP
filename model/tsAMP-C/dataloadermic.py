import os
import random
import torch
import pandas as pd
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
