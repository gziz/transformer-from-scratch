import copy
import torch.nn as nn

import transformer.loaders as loaders
from transformer.utils import Generator
from transformer.embeddings import Embeddings
from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.encoder_decoder import EncoderDecoder
from transformer.ffn import PositionwiseFeedForward
from transformer.mha import MultiHeadedAttention


def make_model(
    src_vocab: int, tgt_vocab: int, N: int=6, d_model: int=512,
    d_ff: int=2048, h: int=8, dropout: float=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator = Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == "__main__":
    spacy_de, spacy_en = loaders.load_tokenizers()
    src_vocab, tgt_vocab = loaders.load_vocab(spacy_de, spacy_en)
    model = make_model(len(src_vocab), len(tgt_vocab), N=6)

# src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    # for each of the args in nn.Sequential, they will get chained
    # i.e.  x  ->  embedding(x)  ->  position(x)  ->  x
