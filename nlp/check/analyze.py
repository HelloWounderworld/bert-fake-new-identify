import numpy as np
import pandas as pd
import sys
import os

path_adjust_parameter = os.path.abspath(os.path.join(os.path.dirname(__file__), '../adjust_parameter'))
sys.path.append(path_adjust_parameter)
import adjust_parameter as ap

path_trainning = os.path.abspath(os.path.join(os.path.dirname(__file__), '../trainning'))
sys.path.append(path_trainning)
from trainning import predict

def parsing(text_parsing=''):

    text_to_parsing = {
        "title": [""],
        "text":[text_parsing],
        "subject": [""],
        "date": [""]
    }

    df = pd.DataFrame(text_to_parsing)

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

    text_to_predict = df_test['text'].iloc[0]

    return predict(text_to_predict)