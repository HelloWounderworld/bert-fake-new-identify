import torch
import numpy as np
from transformers import BertTokenizer

import sys
import os

path_constants = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants'))
sys.path.append(path_constants)
from constants import LABELS

# model_base_name = "google-bert/bert-base-cased" # By Huggingface
# model_base_name = "/teramatsu/nlp/modelbase/bert-base-cased" # By local, inside container - Absolute path
model_base_name = "/teramatsu/nlp/model_bert_fine_tuned" # Fine-tunined model

tokenizer = BertTokenizer.from_pretrained(model_base_name)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [LABELS[label] for label in df["category"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y