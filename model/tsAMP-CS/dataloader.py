import os
import random
import torch
import pandas as pd
from tqdm import tqdm 
import numpy as np

class DataLoader:
    def __init__(self, positive_dir, target_excel, target_base_dir, noise_level=0.1, augment_times=3):
        self.positive_mean_representations = []
        self.miclabels = []  
        self.target_mapping_df = self.load_target_mapping(target_excel)  
        self.noise_level = noise_level
        self.augment_times = augment_times
        self.load_positive_data(positive_dir, target_base_dir)

    def load_target_mapping(self, target_excel):
        df = pd.read_excel(target_excel, header=0)
        return df  

    def add_noise(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level * tensor.std()
        return tensor + noise

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
                        
                        if target in self.target_mapping_df.iloc[:, 1].values:  
                            mic_label = target_row.iloc[0, 6]  
                            self.load_target_data(target, data['mean_representations'], mic_label)

    def load_target_data(self, target_label, positive_mean_representations, mic_label):
        target_folder = '/data2/lhmData/tsAMP/data/target_strains/'
        target_label = target_label.replace(" ","_")
        
        target_path = os.path.join(target_folder, target_label)
        
        if os.path.isdir(target_path):
            target_files = [f for f in os.listdir(target_path) if f.endswith('.pt')]
            
            for target_file in target_files:
                target_data = torch.load(os.path.join(target_path, target_file))
                for target_representation in target_data['mean_representations'].values():
                    for mean_representation in positive_mean_representations.values():
                        combined_vector = torch.cat((mean_representation, target_representation))
                        combined_vector1 = torch.cat((target_representation, mean_representation))
                       
                        self.positive_mean_representations.append(combined_vector)
                        self.miclabels.append(mic_label)

                        for _ in range(self.augment_times):
                            noisy_vector = self.add_noise(combined_vector)
                            self.positive_mean_representations.append(noisy_vector)
                            self.miclabels.append(mic_label)
                            
                            noisy_vector = self.add_noise(combined_vector)
                            self.positive_mean_representations.append(noisy_vector)
                            self.miclabels.append(mic_label)

                            dropout_prob = 0.1 
                            dropout_mask = (torch.rand(combined_vector.shape) > dropout_prob).float()
                            dropout_vector = combined_vector * dropout_mask
                            self.positive_mean_representations.append(dropout_vector)
                            self.miclabels.append(mic_label)

                            scale_factor = random.uniform(0.9, 1.1) 
                            scaled_vector = combined_vector * scale_factor
                            self.positive_mean_representations.append(scaled_vector)
                            self.miclabels.append(mic_label)

                            if len(positive_mean_representations) > 1:
                                other_mean_representation = random.choice(list(positive_mean_representations.values()))
                                mixup_lambda = random.uniform(0.2, 0.8)
                                mixup_vector = mixup_lambda * combined_vector + (1 - mixup_lambda) * torch.cat((other_mean_representation, target_representation))
                                self.positive_mean_representations.append(mixup_vector)
                                self.miclabels.append(mic_label)

    def get_data(self):
        return self.positive_mean_representations, self.miclabels