import torch.nn as nn
import torch

class TSAMPCS(nn.Module):
    def __init__(self):
        super(MIC_Transformer1, self).__init__()  
        self.fc1 = nn.Linear(1280, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)     
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc4(x)  
        return x