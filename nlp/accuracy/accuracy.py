import numpy as np
import pandas as pd
import sys
import os

path_adjust_parameter = os.path.abspath(os.path.join(os.path.dirname(__file__), '../adjust_parameter'))
sys.path.append(path_adjust_parameter)
import adjust_parameter as ap

path_trainning = os.path.abspath(os.path.join(os.path.dirname(__file__), '../trainning'))
sys.path.append(path_trainning)
import trainning as evaluate

def test_accuracy():
    df_fake = pd.read_csv("./fake-and-real-news-dataset/Fake.csv")
    df_true = pd.read_csv("./fake-and-real-news-dataset/True.csv")
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

    print()
    print("Evaluate the content learned!")
    print()

    return evaluate(df_test)
