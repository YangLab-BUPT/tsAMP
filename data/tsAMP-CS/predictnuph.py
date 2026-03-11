import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from micmodel import MIC_Transformer
import numpy as np

import os
import pandas as pd
import torch
from tqdm import tqdm

class DataLoader:
    def __init__(self, positive_dir, taxid_csv):
        self.positive_mean_representations = []
        self.miclabels = []
        self.positive_name = []
        self.target_name = []
        self.target_ids = []
        self.taxid_mapping = self.load_taxid_mapping(taxid_csv)
        self.load_positive_data(positive_dir)

    def load_taxid_mapping(self, taxid_csv):
        df = pd.read_csv(taxid_csv)
        taxid_mapping = df.set_index(df.columns[0]).to_dict()[df.columns[1]]
        
        return {key.replace(' ', ' '): value for key, value in taxid_mapping.items()} 

    def load_positive_data(self, positive_dir):
        positive_files = os.listdir(positive_dir)

        for file in tqdm(positive_files, desc="Loading positive data", unit="file"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(positive_dir, file))
                label = file[:-3]  
                mean_representation = data['mean_representations']  # Assuming this is a dict

                # Extract tensor values
                for representation in mean_representation.values():
                    self.positive_mean_representations.append(representation)  # Append tensors
                    self.positive_name.append(label)

    def get_data(self):
        return self.positive_mean_representations, self.positive_name


def find_target_taxids(model_path, taxid_mapping):
    # Extract the target name from the model path
    target_name = os.path.basename(os.path.dirname(model_path))  # Get the folder name
    print(f"Extracted target name: {target_name}")  # Debugging line
    target_name=target_name.split("_")[1]
    print(target_name)
    # Use the first two words to create the prefix
    target_prefix = ' '.join(target_name.split()[:2])  # Combine first two words
    print(target_prefix)
    print(f"Target prefix: {target_prefix}")  # Debugging line

    # Use a set to avoid duplicate tax IDs
    target_taxids = set()
    
    # Collect tax IDs where the key starts with the target_prefix
    for key, value in taxid_mapping.items():
        if key.startswith(target_prefix):
            target_taxids.add(value)  
    return list(target_taxids)  # Convert set back to list if needed


def predict_and_save_results(model, dataloader, model_path, output_xlsx):
    positive_mean_representations, positive_names = dataloader.get_data()
    
    results = []
    
    # Find the corresponding tax IDs for the model path
    target_taxids = find_target_taxids(model_path, dataloader.taxid_mapping)
    target_taxids1 = list(target_taxids)  # Ensure it's a list
    
    for i, positive_name in enumerate(positive_names):
        for taxid in target_taxids1:
            target_folder = os.path.join('/data2/lhmData/AMP/dataprocess/ESM1v/ncbitarget', str(taxid).strip())
            if os.path.isdir(target_folder):
                target_files = [f for f in os.listdir(target_folder)[:1] if f.endswith('.pt')]
                if target_files:
                    target_data = torch.load(os.path.join(target_folder, target_files[0]))
                    target_representation = target_data['mean_representations']
                    target_tensor = list(target_representation.values())[0]  # Get the first tensor
                    
                    # Combine positive and target representations
                    combined_representation = torch.cat((positive_mean_representations[i], target_tensor)).unsqueeze(0)  # Add batch dimension
                    
                    # Model prediction
                    model.eval()
                    with torch.no_grad():
                        output = model(combined_representation)
                        probabilities = torch.softmax(output, dim=1)
                        _, predictions = torch.max(output, 1)
                        
                   
                    
                    # Append results as a dictionary
                    results.append({
                        'positive': positive_name,
                        'target': target_folder,
                        'taxid': taxid,
                        'predicted_mic': predictions.cpu().numpy()
                    })

    # Save results to Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_xlsx, index=False)  # Save as Excel file
    print(f"Results saved to: {output_xlsx}")

parser = argparse.ArgumentParser(description='Predict MIC labels and save results to CSV.')
parser.add_argument('--output_csv', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--positive_dir', type=str,default='/data2/lhmData/AMP/test/AMP')
parser.add_argument('--taxid_csv', type=str, default='/data2/lhmData/ncbi/taxids.csv')
args = parser.parse_args()

data_loader = DataLoader(args.positive_dir, args.taxid_csv)

model = MIC_Transformer()
model.load_state_dict(torch.load(args.model_path))  
model.eval()

predict_and_save_results(model, data_loader, args.model_path, args.output_csv)