import math
import torch

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot Product Attention
    Args:
        Encoder
            query: (N, H, S, d_k)
            key: (N, H, S, d_k)
            value: (N, H, S, d_v)
            mask: (N, 1, 1, S)
        Decoder 1st MHA
            query:
                T: (N, H, S-1, M), I: (N, H, curr_seq, M)
            key:
                T: (N, H, S-1, K), I: (N, H, curr_seq, K)
            value:
                T: (N, H, S-1, V), I: (N, H, curr_seq, V)
            mask: 
                T: (N, H, S-1, S-1), I: (N, 1, curr_seq, curr_seq)
        Decoder 2nd MHA (Encoder - Decoder)
            query:
                T: (N, H, S-1, M), I: (N, H, curr_seq, M)
            key:
                T: (N, H, S, K), I: (N, H, S, K)
            value:
                T: (N, H, S, V), I: (N, H, S, V)
            mask: 
                T: (N, 1, 1, S), I: (N, 1, curr_seq, S) ???
    """
    d_k = query.size(-1)

    # (N, H, S, d_k) @ (N, H, d_k, S) -> (N, H, S, S)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    ## Encoder: Mask the tokens that are pad
    ## Decoder: Avoid seeing future tokens
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # (N, H, S, S) @ (N, H, S, d_v) -> (N, H, S, d_v)
    return torch.matmul(p_attn, value), p_attn


# Returned values:
# 1. Represents an updated value matrix.
#   The output is a value matrix that's more aware of its surroundings. (we get this by multiplying it with p_attn)
# 2. p_attn: Attention weights, i.e. how much each word should pay attention to others.