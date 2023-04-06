from .base import BaseModel
from .bert_modules.bert import BERT

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):

        batch_size = x.size(0)

        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
class BERTEmbedding(nn.Module):


    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):

        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)


        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        

        self.dropout = nn.Dropout(p=dropout)

        self.embed_size = embed_size

    def forward(self, sequence):

        x = self.token(sequence) + self.position(sequence)  

        return self.dropout(x)

class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.bert_hidden_units = args.bert_hidden_units
        vocab_size = args.num_items + 2
        self.embeddings = nn.ModuleList([BERTEmbedding(vocab_size=vocab_size, embed_size=args.bert_hidden_units, max_len=args.bert_max_len, dropout=args.bert_dropout) for _ in range(self.args.N)])

        self.berts = nn.ModuleList([BERT(args) for _ in range(self.args.N)])
        self.item_classfiers = nn.ModuleList([nn.Linear(self.bert_hidden_units, args.num_items + 1) for _ in range(self.args.N)])

        self.attr_classifiers = nn.ModuleList([nn.Linear(self.bert_hidden_units, args.num_attributes+1) for _ in range(self.args.N)])
        
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, output_type = 'token'):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        logits_list = []
        hiddens_list = []
        for i in range(self.args.N):
            seq = self.embeddings[i](x)
            hidden = self.berts[i](seq, mask)
            hiddens_list.append(hidden)
            if output_type == 'token':
                logits_list.append(self.item_classfiers[i](hidden))
            elif output_type == 'attributes':
                logits_list.append(self.attr_classifiers[i](hidden))
        return logits_list, hiddens_list




