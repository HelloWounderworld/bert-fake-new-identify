from torch import nn
from transformers import BertModel

import sys
import os

path_constants = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants'))
sys.path.append(path_constants)
from constants import LABELS

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_base_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            768, len(LABELS)
        )  # label数に応じて出力先のノード数を変える
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer