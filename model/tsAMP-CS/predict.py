import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from model import TSAMPCS
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr



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
                        
                        if target in self.target_mapping_df.iloc[:, 1].values:  
                            mic_label = target_row.iloc[0, 6]  
                            self.load_target_data(label, target, data['mean_representations'], mic_label)

    def load_target_data(self, label, target_label, positive_mean_representations, mic_label):
        target_folder = '/data2/lhmData/tsAMP/data/target_strains/'
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

def predict_and_save_results(model, dataloader, output_excel, experiment_name=None):
    positive_mean_representations, miclabels, target_name, positive_name = dataloader.get_data()
    metrics_csv = 'result_amp.csv'
    inputs = torch.stack(positive_mean_representations)
    miclabels_tensor = torch.tensor(miclabels)
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    outputs_np = outputs.cpu().numpy().flatten()
    true_labels = miclabels_tensor.cpu().numpy()
    

    final_preds = outputs_np
    final_true = true_labels
    final_positive = positive_name
    final_target = target_name
    
    rounded_preds = np.round(final_preds, 2)
    
    mse = mean_squared_error(final_true, rounded_preds)
    r2 = r2_score(final_true, rounded_preds)
    pearson_corr, pearson_p = pearsonr(final_true, rounded_preds)
    spearman_corr, spearman_p = spearmanr(final_true, rounded_preds)
    
    results = {
        'positive': final_positive,
        'target': final_target,
        'predicted_label': rounded_preds,
        'true_label': final_true
    }
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel, index=False)
    
    metrics_data = {
        'Experiment': experiment_name if experiment_name else os.path.basename(output_excel),
        'MSE': mse,
        'R2': r2,
        'Pearson': pearson_corr,
        'Pearson_p': pearson_p,
        'Spearman': spearman_corr,
        'Spearman_p': spearman_p,
        'Num_Samples': len(final_true)
    }
    
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv)
        df_metrics = pd.concat([df_metrics, pd.DataFrame([metrics_data])], ignore_index=True)
    else:
        df_metrics = pd.DataFrame([metrics_data])
    
    df_metrics.to_csv(metrics_csv, index=False)
    
    print(f"Metrics saved to: {metrics_csv}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}, Pearson: {pearson_corr:.4f} (p={pearson_p:.3g}), Spearman: {spearman_corr:.4f} (p={spearman_p:.3g})")
    
    return metrics_data

parser = argparse.ArgumentParser(description='Predict MIC labels and save results to Excel.')
parser.add_argument('--output_excel', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--test_dir', type=str)
args = parser.parse_args()

positive_dir = '/data2/lhmData/tsAMP/data/AMPesm1v/'
target_base_dir = '/data2/lhmData/tsAMP/data/tsAMP-CS/target_strains/'
target_excel = args.test_dir

data_loader = DataLoader(positive_dir, target_excel, target_base_dir)

model = TSAMPCS()
model.load_state_dict(torch.load(args.model_path))  
model.eval()

predict_and_save_results(model, data_loader, args.output_excel)

