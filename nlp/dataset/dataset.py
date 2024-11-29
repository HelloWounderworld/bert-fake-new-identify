import torch
import numpy as np
from transformers import BertTokenizer

# model_base_name = "google-bert/bert-base-cased" # By Huggingface
model_base_name = "/home/seu_usuario/modelbase/bert-base-cased" # By local, inside container - Absolute path

tokenizer = BertTokenizer.from_pretrained(model_base_name)
labels = {"Fake": 0, "Real": 1}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df["category"]]
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