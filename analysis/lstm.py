# analysis/lstm.py
import torch
import torch.nn as nn

class HAR_LSTM(nn.Module):
    def __init__(self, n_features, seq_length, n_hidden, n_layers):
        super(HAR_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.n_hidden,
            num_layers = self.n_layers,
            batch_first = True
        )

        self.linear = nn.Linear(
            self.n_hidden * self.seq_length,
            1
        )

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(
            self.n_layers,
            batch_size,
            self.n_hidden
        )
        cell_state = torch.zeros(
            self.n_layers,
            batch_size,
            self.n_hidden
        )
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.linear(x)



