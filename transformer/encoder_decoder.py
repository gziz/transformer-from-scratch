import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args
            src: 
                T,I: (N, S)
            tgt:  
                T: (N, S-1), I: (N, curr_seq)
            src_mask:
                T,I: (N, 1, S)
            tgt_mask: 
                T: (N, curr_seq, curr_seq), I: (N, curr_seq, curr_seq)
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
            src: 
                T,I: (N, S)
            src_mask:
                T,I: (N, 1, S)
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Args:
            memory: 
                T, I: (N, S, M)
            src_mask:
                T, I: (N, 1, S)
            tgt:
                T: (N, S-1), I: (N, curr_seq)
            tgt_mask: 
                T: (N, S-1, S-1), I: (N, curr_seq, curr_seq)
        Returns:

        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
            # (x, memory, src_mask, tgt_mask)
