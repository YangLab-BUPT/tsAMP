import torch.nn as nn
import torch

class TSAMPC(nn.Module):
    def __init__(self):
        super(TSAMPC, self).__init__()  
        self.fc1 = nn.Linear(2560, 256)
        self.dropout = nn.Dropout(0.3)  
        #self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.3)  
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)  
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        #x = torch.relu(self.fc2(x))
        #x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x 
        
        
