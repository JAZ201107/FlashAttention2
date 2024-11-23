import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, mask):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


def scaled_dot_product_attention(Q, K, V, softmax_scale, mask):
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.MODEL.d_model
        self.h = config.MODEL.heads

        self.flash_attention = config.MODEL.flash_attention

        assert (
            self.d_model % self.h == 0
        ), "d_model must be divisible by the number of heads"

        self.d_k = self.d_model // self.h

        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.MODEL.dropout)

    def forward(self, q, k, v, mask):
        B = q.size(0)

        q, k, v = self.q(q), self.k(k), self.v(v)

        q = q.view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, -1, self.h, self.d_k).transpose(1, 2)

        if not self.flash_attention:
            x = scaled_dot_product_attention(q, k, v, self.d_k**0.5, mask)
        else:
            x = FlashAttention.apply(q, k, v, self.d_k**0.5, mask)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.out(x)
