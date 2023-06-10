import torch
from torch import nn
import torch.nn.functional as F
import config

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).to(config.device)
        self.norm = torch.nn.LayerNorm(hidden_size).to(config.device)
        self.fc2 = nn.Linear(hidden_size, output_size).to(config.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
