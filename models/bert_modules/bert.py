from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()


        fix_random_seed_as(args.model_init_seed)



        max_len = args.bert_max_len
        num_items = args.num_items

        n_layers = args.bert_num_blocks

        heads = args.bert_num_heads
        vocab_size = num_items + 2

        hidden = args.bert_hidden_units
        self.hidden = hidden

        dropout = args.bert_dropout

        #self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):

        #x = self.embedding(x)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
