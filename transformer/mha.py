import torch.nn as nn

from transformer.utils import clones
from transformer.attention import attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, M, dropout=0.1):
        """Take in model size and number of heads.
        Args:
            h: number_of_heads
        """
        super(MultiHeadedAttention, self).__init__()
        assert M % h == 0
        # We assume d_v always equals d_k
        self.d_k = M // h
        self.h = h
        self.linears = clones(nn.Linear(M, M), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            When using MHA for Encoder:
                q: (N, S, M)
                k: (N, S, K)
                v: (N, S, V)
                mask: (N, 1, S)
            Decoder 1st MHA
                q: 
                    T: (N, S-1, M), I: (N, curr_seq, M)
                k: 
                    T: (N, S-1, M), I: (N, curr_seq, K)
                v: 
                    T: (N, S-1, M), I: (N, curr_seq, V)
                mask: 
                    T: (N, S-1, S-1), I: (N, curr_seq, curr_seq)
            Decoder 2nd MHA (Encoder - Decoder)
                q: 
                    T: (N, S-1, M), I: (N, curr_seq, M)
                k: 
                    T: (N, S, K), I: (N, S, K)
                v:
                    T: (N, S, V), I: (N, S, V)
                mask: 
                    T: (N, 1, S), I: (N, 1, S)
        * Where
            K: Key size, V: Value size
                Note: K, V are almost always equal to M
            * : Represents the number of tokens.
        Returns:
            x: (N, S, d_model)
                where x is the value matrix weighted by attention
                (how much each token i should pay attention to j)
        """
        if mask is not None:
            # Same mask applied to all heads.
            # (N, M) -> (N, 1, M)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch.
        # lin(): (N, *, M) -> (N, *, M)
        # view().transpose(): (N, *, M) -> (N, H, *, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # Notice that the x (N, H, *, d_k)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view
        # (N, H, *, d_v) -> (N, *, H, d_v) -> (N, *, M)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        # This final linear corresppnds to W^O
        # (N, S, M) -> (N, S, M)
        return self.linears[-1](x)