import numpy as np 
import pandas as pd

import adjust_parameter as ap
from trainning.trainning import train, evaluate, BertClassifier

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

EPOCHS = 4
model = BertClassifier()
LR = 1e-6     

print("Trainning result: ", train(model, df_train, df_val, LR, EPOCHS))

evaluate(model, df_test)
