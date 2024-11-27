import numpy as np 
import pandas as pd

import adjust_parameter as ap
from trainning.trainning import load_model, predict, check_reliability, BertClassifier

df_fake = pd.read_csv("Fake.csvのパス")
df_true = pd.read_csv("True.csvのパス")
df_true['category'] = "Real" #カテゴリーを追加
df_fake['category'] = "Fake" #カテゴリーを追加
df = pd.concat([df_true, df_fake]).reset_index(drop = True) #2つのデータセットを統合

print()
print("Verify inside: ", df.head())
print()

df['text'] = df['text'].str.lower() #すべて小文字にする
df['text'] = df['text'].apply(lambda text: ap.remove_URL(text))
df['text'] = df['text'].apply(lambda text: ap.remove_html(text))
df['text'] = df['text'].apply(lambda text: ap.remove_atsymbol(text))
df['text'] = df['text'].apply(lambda text: ap.remove_hashtag(text))
df['text'] = df['text'].apply(lambda text: ap.remove_exclamation(text))
df['text'] = df['text'].apply(lambda text: ap.remove_punc(text))
df['text'] = df['text'].apply(lambda text: ap.remove_number(text))
df['text'] = df['text'].apply(lambda text: ap.remove_emoji(text))

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])

model = BertClassifier()
save_path = "/home/seu_usuario/model_bert_fine_tuned" 

loaded_model, loaded_tokenizer = load_model('/home/seu_usuario/model_bert_fine_tuned', model)

# Agora você pode usar o modelo carregado para fazer previsões
# Por exemplo, em uma função de inferência
text_to_predict = "Seu texto aqui"
print(check_reliability(model, loaded_tokenizer, text_to_predict))

result = predict(loaded_model, df_test)
print(result)
