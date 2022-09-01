import torch
from torch import nn

class Simple1DCNN(nn.Module):
    def __init__(self, D, K, use_dropout=False):
        super(Simple1DCNN, self).__init__()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.25)
        self.layer1 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=1, stride=2)
        self.fc_output = nn.Linear(100, 50)
        self.fc_output2 = nn.Linear(50, K)

    def forward(self, x):
        x = x[:,:,None]
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        x = self.layer2(x)
        # print(x.shape)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        x = torch.squeeze(x)
        x = self.fc_output(x)
        x = self.fc_output2(x)

        return x