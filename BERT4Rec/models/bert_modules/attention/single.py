import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        """
        dimension of query, key and value is batch_size x head_num x max_len x (hidden_size // head_num)

        scores: batch_size x head_num x max_len x max_len

        return: batch_size x head_num x max_len x hidden_size(value)
        """

        # print("query size: ", query.size())
        # print("scores' size: ", scores.size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
