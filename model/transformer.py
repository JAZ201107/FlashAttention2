import torch.nn as nn


from blocks import *


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.input_embed = nn.Embedding(
            config.DATA.source_vocab_size, config.MODEL.d_model
        )
        self.output_embed = nn.Embedding(
            config.DATA.target_vocab_size, config.MODEL.d_model
        )

        self.src_pos = PositionalEncoding(config.MODEL.source_seq_len)
        self.tgt_pos = PositionalEncoding(config.MODEL.target_seq_len)

        self.projection = ProjectionLayer(config)

        # Weight typing
        self.output_embed.weight = self.projection.linear.weight

        self.apply(Transformer._init_weight)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.input_embed(src) + self.src_pos(src)
        tgt = self.output_embed(tgt) + self.tgt_pos(tgt)

        context = self.encoder(src, src_mask)
        output = self.decoder(tgt, context, src_mask, tgt_mask)

        return self.projection(output)

    @staticmethod
    def _init_weight(module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
