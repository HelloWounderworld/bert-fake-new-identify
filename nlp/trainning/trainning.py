import torch
from torch.optim import Adam
from tqdm import tqdm

import sys
import os

path_datasetclass = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../datasetclass")
)
sys.path.append(path_datasetclass)
from datasetclass import Dataset, tokenizer

path_modelclass = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../modelclass")
)
sys.path.append(path_modelclass)
from modelclass import BertClassifier

path_constants = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../constants")
)
sys.path.append(path_constants)
from constants import LABELS


def train(
    train_data,
    val_data,
    learning_rate,
    epochs,
    save_path,
):
    model = BertClassifier()

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


def evaluate(test_data):
    model = BertClassifier()

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

    # print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")
    return {"test_accuracy": round(total_acc_test / len(test_data), 3)}


def predict(text):
    model = BertClassifier()

    checkpoint = torch.load("/teramatsu/bert-nlp/moel_bert_fine_tuned/checkpoint.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        inputs = loaded_tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        mask = inputs["attention_mask"]
        input_id = inputs["input_ids"].squeeze(1)

        # Obtém os logits (scores antes da classificação)
        output = model(input_id, mask)

        # Calcula as probabilidades usando softmax
        probabilities = torch.softmax(output, dim=1)

        # Obtém a predição (classe com maior probabilidade)
        prediction = torch.argmax(output, dim=1)

        return {
            "class": list(LABELS.keys())[prediction.item()],
            "scores": output.squeeze().tolist(),
            "probabilities": {
                "Fake": probabilities.squeeze().tolist()[0],
                "Real": probabilities.squeeze().tolist()[1],
            },
        }
