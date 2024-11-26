# Carregar modelo
model = BertClassifier()
model.load_state_dict(torch.load('./../model_bert_fine_tuned/checkpoint.pth')['model_state_dict'])

# Ou usar o método da Hugging Face
from transformers import BertModel, BertTokenizer

model = BertClassifier()
model.bert = BertModel.from_pretrained('./../model_bert_fine_tuned')
tokenizer = BertTokenizer.from_pretrained('./../model_bert_fine_tuned')