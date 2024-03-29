import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocessing(sentence):
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())

    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    tokenized_sentence = word_tokenize(sentence)

    return tokenized_sentence

def calculate_difficulty(df):
    scaler = StandardScaler()
    df['z_scores'] = scaler.fit_transform(pd.DataFrame(df['count'].values))
    minmax_scaler = MinMaxScaler()
    df['difficulties'] = minmax_scaler.fit_transform(df['z_scores'].values.reshape(-1, 1))

    return df

def calculate_chapter_difficulty(chapter, difficulty):

    word_difficulty = dict(zip(difficulty['word'], difficulty['difficulties']))

    difficulties = [word_difficulty.get(word, 0) for word in chapter]
    if difficulties:
        return sum(difficulties) / len(difficulties)
    else:
        return 0

df_train = pd.read_parquet('output/train.parquet')
df_test = pd.read_parquet('output/test.parquet')
df_val = pd.read_parquet('output/validation.parquet')
df_difficulty = pd.read_csv('output/ngram_freq.csv')

df_concat = pd.concat([df_train, df_test, df_val], axis=0)

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: x.encode('latin1', errors='ignore').decode('utf-8'))
df_concat['book_id'] = df_concat['book_id'].str.lower()
df_concat['book_id'] = df_concat['book_id'].str.replace(",", "")
df_concat['book_id'] = df_concat['book_id'].str.replace(":", "")
df_concat['book_id'] = df_concat['book_id'].str.replace("\\", "")

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: re.search(r'^(.*?)\.', x).group(1) if re.search(r'^(.*?)\.', x) else x)
df_concat = df_concat.drop_duplicates(subset=['book_id'], keep='first')
df_concat = df_concat[['book_id', 'chapter']]

df_difficulty = calculate_difficulty(df_difficulty)
df_concat['tokenized_chapter'] = df_concat['chapter'].apply(preprocessing)
# df_concat['chapter_difficulty'] = calculate_chapter_difficulty(df_concat['tokenized_chapter'], df_difficulty)


a = 5
