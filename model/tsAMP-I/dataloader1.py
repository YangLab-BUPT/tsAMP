import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
#def __init__(self, positive_dir, negative_dir, negative3_dir, negative1_dir, negative2_dir, 
                 #excel_file, excel_file2, excel3_file2, excel1_file2, excel2_file2, num_augmentation=3):
class DataLoader:
    def __init__(self, positive_dir, negative_dir, negative3_dir, negative1_dir, negative2_dir, 
                 excel_file, excel_file2, excel3_file2, excel1_file2, excel2_file2, num_augmentation=3):
        self.data = []
        self.labels = []
        self.num_augmentation = num_augmentation  # 需要生成的增强样本数量

        self.physical_properties_positive = self.load_physical_properties(excel_file)
        self.physical_properties_negative = self.load_physical_properties(excel_file2)
        self.physical_properties_negative1 = self.load_physical_properties(excel1_file2)
        self.physical_properties_negative2 = self.load_physical_properties(excel2_file2)
        self.physical_properties_negative3 = self.load_physical_properties(excel3_file2)

        self.load_samples(positive_dir, label=1, physical_properties=self.physical_properties_positive)
        self.load_samples(negative_dir, label=0, physical_properties=self.physical_properties_negative)
        self.load_samples(negative1_dir, label=0, physical_properties=self.physical_properties_negative1, limit=100000)
        self.load_samples(negative2_dir, label=0, physical_properties=self.physical_properties_negative2, limit=2000)
        self.load_samples(negative3_dir, label=0, physical_properties=self.physical_properties_negative3, limit=1)

    def load_samples(self, data_dir, label, physical_properties, limit=None):
        files = os.listdir(data_dir)
        if limit is not None:
            files = files[:limit]  # Limit the number of files to read if specified

        for file in tqdm(files, desc=f"Loading {'positive' if label == 1 else 'negative'} samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(data_dir, file))
                mean_representation = torch.stack(list(data['mean_representations'].values()))
                mean_representation_flat = mean_representation.flatten()

                sample_label = file[:-3] 
                if sample_label in physical_properties.index:
                    props = physical_properties.loc[sample_label].values[1:5]
                    #print(props)
                    props = torch.tensor(props.astype(float), dtype=torch.float32)
                    
                    # 生成原始样本的结合表示
                    combined_representation = torch.cat((mean_representation_flat, props), dim=0)
                    self.data.append(combined_representation)
                    self.labels.append(label)
                    # 生成多个增强样本
                    #for _ in range(self.num_augmentation):
                        # 数据增强：添加噪声
                        #noise = torch.randn(combined_representation.size()) * 0.01  # 添加小的噪声
                        #augmented_representation = combined_representation + noise

                        #self.data.append(augmented_representation)
                        #self.labels.append(label)

    '''def load_random_negative_samples(self, data_dir, label, physical_properties, limit=None):
        files = os.listdir(data_dir)
        if limit is not None:
            files = random.sample(files, min(len(files), limit))  # 随机选择文件

        for file in tqdm(files, desc=f"Loading random {'negative'} samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(data_dir, file))
                mean_representation = torch.stack(list(data['mean_representations'].values()))
                mean_representation_flat = mean_representation.flatten()

                sample_label = file[:-3] 
                if sample_label in physical_properties.index:
                    props = physical_properties.loc[sample_label].values[1:5]
                    print(props)
                    props = torch.tensor(props.astype(float), dtype=torch.float32)

                    combined_representation = torch.cat((mean_representation_flat, props), dim=0)
                    self.data.append(combined_representation)
                    self.labels.append(label)
                    # 生成多个增强样本
                    for _ in range(self.num_augmentation):
                        # 数据增强：添加噪声
                        noise = 0  # 添加小的噪声
                        augmented_representation = combined_representation + noise

                        self.data.append(augmented_representation)
                        self.labels.append(label)'''

    def load_physical_properties(self, excel_file):
        df = pd.read_excel(excel_file)
        return df.set_index(df.columns[0]) 

    def get_data(self):
        if not self.data:
            print("Error: No samples found.")
            return None, None, None, None

        X = torch.stack(self.data)
        y = torch.tensor(self.labels, dtype=torch.float32)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_val, y_train, y_val