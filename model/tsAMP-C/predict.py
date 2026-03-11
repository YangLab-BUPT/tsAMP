import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from micmodel import TSAMPC
import numpy as np
import csv
class SimpleDataLoader:
    def __init__(self, positive_dir):
        self.positive_data = [] 
        self.load_positive_data(positive_dir)

    def load_positive_data(self, positive_dir):
        positive_files = [f for f in os.listdir(positive_dir) if f.endswith('.pt')]
        for file in tqdm(positive_files, desc="Loading positive data"):
            file_path = os.path.join(positive_dir, file)
            data = torch.load(file_path)
            for seq_id, representation in data['mean_representations'].items():
                self.positive_data.append({
                    'name': f"{file}_{seq_id}",
                    'vector': representation
                })

    def get_data(self):
        return self.positive_data

def predict_and_save_results(model, positive_list, target_pt_path, output_xlsx):
    print(f"Loading fixed target: {target_pt_path}")
    target_data = torch.load(target_pt_path)
    target_vector = list(target_data['mean_representations'].values())[0]
    
    results = []
    model.eval()

    with torch.no_grad():
        for item in tqdm(positive_list, desc="Predicting"):
            pos_vector = item['vector']
            pos_name = item['name']
            
            combined_representation = torch.cat((pos_vector, target_vector)).unsqueeze(0)
            
            output = model(combined_representation)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            print(prediction)
            results.append({
                'peptide_name': pos_name,
                'target_species': 'Candida albicans',
                'target_file': os.path.basename(target_pt_path),
                'predicted_mic_label': prediction
            })


    df_results = pd.DataFrame(results)
    df_results.to_excel(output_xlsx, index=False)
    print(f"Results saved to: {output_xlsx}")


parser = argparse.ArgumentParser(description='Predict MIC labels with fixed target.')
parser.add_argument('--output_excel', type=str, default='results.xlsx')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--test_dir', type=str, help='Directory containing positive .pt files')
args = parser.parse_args()

positive_dir = '/tsAMP/data/AMPesm1v'
fixed_target_path = "/tsAMP/data/tsAMP-C/species/Candida_albicans.pt"


data_loader = SimpleDataLoader(positive_dir)
positive_data_list = data_loader.get_data()

model = TSAMPC()
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
model.eval()

predict_and_save_results(model, positive_data_list, fixed_target_path, args.output_excel)

