import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # [:, 0::2] -> comenzando en el 0, moviendo en saltos de 2.
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        #pe: (1, d_model, max_seq)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_sz, seq_len, d_model)
        # get up until seq_len from the max_seq dimension.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)