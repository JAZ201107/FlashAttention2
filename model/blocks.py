import torch
import torch.nn as nn
import torch.nn.functional as F

from normalization import LayerNorm
from attention import MultiHeadAttention


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(config.MODEL.d_model, config.MODEL.d_ff)
        self.dropout = nn.Dropout(config.MODEL.dropout)
        self.linear_2 = nn.Linear(config.MODEL.d_ff, config.MODEL.d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.MODEL.d_model
        self.vocab_size = config.DATA.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model**0.5)


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.MODEL.d_model
        self.seq_len = config.DATA.seq_len
        self.dropout = nn.Dropout(config.MODEL.dropout)

        # Create a matrix of shape (seq_len, self.d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        # Create a vector of length of seq_len, which indicates the position of the word
        position = torch.arange(0, self.seq_len).unsqueeze(1).float()

        # Create a vector of shape (d_model) with even indices
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.size(1) is the sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()

        self.dropout = config.MODEL.dropout
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForwardBlock(config)
        self.residual = nn.ModuleList(
            [ResidualConnection(config.MODEL.d_model, config) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = LayerNorm(config.MODEL.d_model)
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.MODEL.encoder_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)

        self.feed_forward = FeedForwardBlock(config)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(config.MODEL.d_model, config) for _ in range(3)]
        )

    def forward(self, x, context, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, context, context, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = LayerNorm(config.MODEL.d_model)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.MODEL.decoder_layers)]
        )

    def forward(self, x, context, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(config.MODEL.d_model, config.DATA.vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
