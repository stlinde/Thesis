# analysis/dnn.py
import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, seq_length):
        super(LinearBlock, self).__init__()
        self.seq_length = seq_length

        self.layer_1 = nn.Linear(self.seq_length, 128, bias=False)
        self.layer_2 = nn.Linear(128, 256, bias=False)
        self.layer_3 = nn.Linear(256, 128, bias=False)
        self.layer_4 = nn.Linear(128, 64, bias=False)
        self.layer_5 = nn.Linear(64, 8, bias=False)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.bn1(x)
        x = self.relu(self.layer_3(x))
        x = self.bn2(x)
        x = self.relu(self.layer_4(x))
        return self.layer_5(x)


class HAR_DNN(nn.Module):
    def __init__(self, n_features, seq_length):
        super(HAR_DNN, self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length

        self.daily_block = LinearBlock(seq_length=32)
        self.weekly_block = LinearBlock(seq_length=32)
        self.monthly_block = LinearBlock(seq_length=32)

        self.layer_1 = nn.Linear(24, 48)
        self.layer_2 = nn.Linear(48, 12)
        self.layer_3 = nn.Linear(12, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        daily = self.daily_block(x[:, 0, :])
        weekly = self.weekly_block(x[:, 1, :])
        monthly = self.monthly_block(x[:, 1, :])

        x = torch.cat((daily, weekly, monthly), dim=1)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return self.layer_3(x)



class HAR_DNN_6L(nn.Module):
    def __init__(self, n_features):
        super(HAR_DNN, self).__init__()
        self.n_features = n_features

        self.layer_1 = nn.Linear(self.n_features, 8, bias=False)
        self.layer_2 = nn.Linear(8, 16, bias=False)
        self.layer_3 = nn.Linear(16, 32, bias=False)
        self.layer_4 = nn.Linear(32, 16, bias=False)
        self.layer_5 = nn.Linear(16, 8, bias=False)
        self.layer_6 = nn.Linear(8, 1)

        self.silu = nn.SiLU()
        self.revin = ReversibleInstanceNormalization()

    def forward(self, x):
        x = self.revin(x, "norm")
        x = self.silu(self.layer_1(x))
        x = self.silu(self.layer_2(x))
        x = self.silu(self.layer_3(x))
        x = self.silu(self.layer_4(x))
        x = self.silu(self.layer_5(x))
        x = self.revin(x, "denorm")
        return self.layer_6(x)

