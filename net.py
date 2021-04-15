from torch import nn
import torch

class DeepQNet(nn.Module):

    def __init__(self, hidden_size=64):
        super(DeepQNet, self).__init__()
        self.state_space_size = 128
        self.action_space_size = 18
        self.hidden_size = hidden_size
        

        self.mlp = nn.Sequential(
            nn.Linear(self.state_space_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_space_size)
        )
        
    def forward(self, x):
        x = (x-128)/128
        return self.mlp(x)