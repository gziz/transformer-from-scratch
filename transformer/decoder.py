import torch.nn as nn

from transformer.utils import clones
from transformer.layers import LayerNorm, SublayerConnection

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x (N, curr_seq, d_model)
            memory: (N, max_seq, d_model)
                The output from the encoder's layer.
        Returns:
            x (N, curr_seq, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn (enc-dec attn), and feed forward"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x:
                T:(N, S-1, M), I:(N, curr_seq_len, M)
            memory:
                T:(N, S, M), I:(N, S, M)
            src_mask:
                T:(N, 1, S), I:(N, 1, S)
            tgt_mask:
                T:(), I:(N, 1, curr_seq_len)
        """

        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)