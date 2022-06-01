# analysis/tcn.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1D, self).__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.kernel_size - 1) * self.dilation

        self.conv = weight_norm(nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation
        ))

    def forward(self, x):
        return self.conv(x)[:, :, :-self.dilation].contiguous()

class CausalConvBlock(nn.Module):
    def __init__(self, seq_length, dilation, dropout=0.2):
        super(CausalConvBlock, self).__init__()
        self.seq_length = seq_length
        self.dilation = dilation
        self.dropout = dropout

        self.layer_1 = CausalConv1D(1, 1, kernel_size=2,
                                                dilation=self.dilation)
        self.layer_2 = CausalConv1D(1, 1, kernel_size=2,
                                                dilation=self.dilation)

        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout2(x)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, seq_length, dropout):
        super(TemporalConvNet, self).__init__()
        self.seq_length = seq_length
        self.dropout = dropout

        self.block_1 = nn.Sequential(
            CausalConvBlock(self.seq_length, dilation=1, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=2, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=4, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=8, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=16, dropout=self.dropout),
        )
        self.block_2 = nn.Sequential(
            CausalConvBlock(self.seq_length, dilation=1, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=2, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=4, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=8, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=16, dropout=self.dropout),
        )
        self.block_3 = nn.Sequential(
            CausalConvBlock(self.seq_length, dilation=1, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=2, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=4, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=8, dropout=self.dropout),
            CausalConvBlock(self.seq_length, dilation=16, dropout=self.dropout),
        )

        self.layer_1 = nn.Linear(96, 256, bias=False)
        self.layer_2 = nn.Linear(256, 128, bias=False)
        self.layer_3 = nn.Linear(128, 32, bias=False) 
        self.layer_4 = nn.Linear(32, 1, bias=False)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        daily = self.block_1(x[:, 0, :].reshape([-1, 1, 32]))
        weekly = self.block_2(x[:, 1, :].reshape([-1, 1, 32]))
        monthly = self.block_2(x[:, 2, :].reshape([-1, 1, 32]))

        x = torch.cat((daily, weekly, monthly), dim=2)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        return self.layer_4(x)



