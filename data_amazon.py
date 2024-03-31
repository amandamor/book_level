import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect

proficiency_intervals = {
    'A1': (0, 0.2),
    'A2': (0.2, 0.4),
    'B1': (0.4, 0.6),
    'B2': (0.6, 0.8),
    'C1': (0.8, 0.95),
    'C2': (0.95, 1.0)
}

def preprocessing(sentence):
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())

    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    stopWords = set(stopwords.words('english'))
    tokenized_sentence = word_tokenize(sentence)

    wordsFiltered = [w for w in tokenized_sentence if w not in stopWords]

    document = ' '.join(wordsFiltered)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())

    X_tfidf = tfidf_vectorizer.fit_transform([document])

    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = X_tfidf.toarray()[0]

    return feature_names, tfidf_values

def calculate_difficulty(df):
    df = df.copy()

    word_counts = df['tokenized_chapter_names'].value_counts()

    df['count'] = df['tokenized_chapter_names'].map(word_counts)

    scaler = StandardScaler()
    df['z_scores'] = scaler.fit_transform(pd.DataFrame(df['count']))

    minmax_scaler = MinMaxScaler()
    df['difficulty'] = minmax_scaler.fit_transform(df['z_scores'].values.reshape(-1, 1))

    return df

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

def map_proficiency(difficulty):
    for proficiency, interval in proficiency_intervals.items():
        if interval[0] <= difficulty < interval[1]:
            return proficiency
    return None

df_json = pd.read_json('./output/final.json')
df = pd.DataFrame(df_json)

print(df.info())
print(df.isna().sum())
print(df.describe())
print(len(df))

df['is_english'] = df['chapter'].apply(is_english)
df = df[df['is_english']]
df = df.drop(columns=['is_english', 'isbn'])

print(len(df))

df['author'] = df['author'].str.lower()
df['book_title'] = df['book_title'].str.lower()
df['category'] = df['category'].str.lower()
df['chapter'] = df['chapter'].str.lower()

df[['tokenized_chapter_names', 'tokenized_chapter_values']] = df['chapter'].apply(lambda x: pd.Series(preprocessing(x)))

df_exploded = df.explode(['tokenized_chapter_names', 'tokenized_chapter_values'])

df_difficulty = calculate_difficulty(df_exploded)

df_difficulty_final = df_difficulty.groupby(['author', 'book_title', 'category'])['difficulty'].mean().reset_index()

df_difficulty_final['english_level'] = df_difficulty_final['difficulty'].apply(map_proficiency)

df_difficulty_final.to_csv('./output/difficulty_final.csv')
