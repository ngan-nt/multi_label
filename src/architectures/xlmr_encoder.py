from torch import nn
import torch
from transformers import XLMRobertaModel, AutoModel
from torch.nn.functional import gelu
import pickle

from pathlib import Path

class XLMREncoder(nn.Module):
    """Simple encoder with custom XLM-Roberta pretrained model.
    """
    def __init__(self, hparams):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(hparams.pretrained_path) # load pretrained model
        self.encoder_size = hparams.bert_output # encoder output size - 768 for base architecture
        self.num_output = hparams.num_output # number of label - 6
        self.fc = nn.Linear(self.encoder_size, self.num_output) # fully connected layer
        self.out = nn.Sigmoid()


    def forward(self, ids):
        # get bert output
        encoder_output = self.encoder(ids)[0]
        encoder_output = encoder_output[:, -1] # get CLS embedding

        # fully connected layer
        output = self.fc(encoder_output)
        output = self.out(output)

        return output


