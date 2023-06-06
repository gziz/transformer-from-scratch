
import torch.nn as nn
from torch import tensor

from transformer.utils import clones
from transformer.layers import LayerNorm, SublayerConnection

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: int, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: tensor, mask: tensor) -> tensor:
        """Pass the input (and mask) through each of self.layers.
         Each layer represents a block of (MHA, FFN).

        :param x: (N, S, M)
        :param mask: (N, 1, S)
        :return: (N, S, M)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: tensor, mask: tensor) -> tensor:
        """
        :param x: (N, S, M)
        :param mask: (N, 1, S)
        :return: (N, S, M)
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

