import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):

    def __init__(self, feat_dim, dropout_rate, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2) * (-math.log(10000.0) / feat_dim))
        pe = torch.zeros(1, max_len, feat_dim, )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def generate_mask( self, x ):
        return x != 0.0

    def forward( self, x ):
        mask = self.generate_mask( x )
        x = x + self.pe[ 0, :x.size(1), :, ]
        #x *= mask
        return self.dropout( x )

