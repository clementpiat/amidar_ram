from torch import nn

class DeepQNet(nn.Module):

    def __init__(self, hidden_size=128):
        super(DeepQNet, self).__init__()
        self.state_space_size = 128
        self.action_space_size = 10
        self.hidden_size = hidden_size
        

        self.mlp = nn.Sequential(
            nn.Linear(self.state_space_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.action_space_size)
        )
        
    def forward(self, x):
        return self.mlp(x)