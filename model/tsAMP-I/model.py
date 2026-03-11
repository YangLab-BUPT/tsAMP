# model.py

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout_rate)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(128, output_dim)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x) 
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)
        
        return x