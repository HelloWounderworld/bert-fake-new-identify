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


from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_base_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            768, len(labels)
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


from torch.optim import Adam
from tqdm import tqdm


def train(
    model,
    train_data,
    val_data,
    learning_rate,
    epochs,
    save_path,
):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

    # Após o loop de treinamento
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss_train,
        },
        f"{save_path}/checkpoint.pth",
    )

    # Salva o modelo e tokenizador
    model.bert.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
    
def load_model(save_path, model):
    # Carrega o tokenizador
    tokenizer = BertTokenizer.from_pretrained(save_path)
    
    # Carrega os pesos do modelo
    checkpoint = torch.load(f'{save_path}/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Coloca o modelo em modo de avaliação
    model.eval()
    
    return model, tokenizer

def predict(model, loaded_tokenizer, text):
    
    model.eval()
    with torch.no_grad():
        inputs = loaded_tokenizer(text, padding='max_length', max_length=512, 
                                  truncation=True, return_tensors="pt")
        mask = inputs['attention_mask']
        input_id = inputs['input_ids'].squeeze(1)
        
         # Obtém os logits (scores antes da classificação)
        output = model(input_id, mask)
        
        # Calcula as probabilidades usando softmax
        probabilities = torch.softmax(output, dim=1)
        
        # Obtém a predição (classe com maior probabilidade)
        prediction = torch.argmax(output, dim=1)
        
        return {
            'class': list(labels.keys())[prediction.item()],
            'scores': output.squeeze().tolist(),
            'probabilities': probabilities.squeeze().tolist()
        }
        
        # Cospe se e Fake ou Real, apenas.
        # output = model(input_id, mask)
        # prediction = torch.argmax(output, dim=1)
        
        # return list(labels.keys())[prediction.item()]

def check_reliability(model, test_data):
    model.eval()
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy, fine-tuned model: {total_acc_test / len(test_data): .3f}")
