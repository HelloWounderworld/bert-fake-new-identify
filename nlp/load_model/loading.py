# Carregar modelo
model = BertClassifier()
model.load_state_dict(torch.load('./modelo_bert_ajustado/checkpoint.pth')['model_state_dict'])

# Ou usar o método da Hugging Face
from transformers import BertModel, BertTokenizer

model = BertClassifier()
model.bert = BertModel.from_pretrained('./modelo_bert_ajustado')
tokenizer = BertTokenizer.from_pretrained('./modelo_bert_ajustado')