from torch import nn


class DenoisingModel(nn.Module):
    """
    ni: number of input neurons (x0, x1, t)
    nh: number of hidden neurons 
    no: number of output neurons
    """
    def __init__(self, ni=3, no=2, nh=256):
        super(DenoisingModel, self).__init__()

        self.il = nn.Linear(ni, nh)
        self.hl = nn.Linear(nh, nh)
        self.ol = nn.Linear(nh, no)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.il(x)
        x = self.tanh(x)
        x = self.hl(x)
        x = self.relu(x)
        x = self.hl(x)
        x = self.relu(x)
        x = self.ol(x)
        return x