import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_shape = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        return self.layers(x)
