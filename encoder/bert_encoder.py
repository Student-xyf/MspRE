import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import json, os


class BERTEncoder(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path, output_hidden_states=True)

    def forward(self, token, att_mask):
        x = self.bert(token, attention_mask=att_mask)
        hidden_states = x.hidden_states

        # Combine all hidden states with weighted averaging
        weights = torch.tensor([1.0 / self.num_layers] * self.num_layers, device=token.device)
        weighted_avg = torch.sum(torch.stack(hidden_states) * weights.view(-1, 1, 1, 1), dim=0)

        return weighted_avg