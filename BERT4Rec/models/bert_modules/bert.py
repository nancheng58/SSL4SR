from torch import nn as nn
import torch
from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        max_len = args.max_len
        self.max_len = max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        # 2 means [mask] (item_num + 1) and padding (0)
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        hidden_dropout = args.bert_hidden_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden,
                                       max_len=max_len, dropout=hidden_dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout, hidden_dropout) for _ in range(n_layers)])

        self.unidirectional_tf_blocks = None

    def forward(self, x):
        # x's dimension: batch_size x max_len
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # batch_size x 1 x max_len x max_len
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x

    def init_weights(self):
        pass
