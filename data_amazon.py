import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader


proficiency_intervals = {
    'A1': (0, 0.2),
    'A2': (0.2, 0.4),
    'B1': (0.4, 0.6),
    'B2': (0.6, 0.8),
    'C1': (0.8, 0.95),
    'C2': (0.95, 1.0)
}
def dividir_em_sentencas(texto):
    sentencas = nltk.sent_tokenize(texto)
    return sentencas

def preprocessing(sentence):
    sentence = ''.join(char for char in sentence if not char.isdigit())

    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    stopWords = set(stopwords.words('english'))
    tokenized_sentence = word_tokenize(sentence)

    wordsFiltered = []

    for sent in sentence:
        tokenized_sentence = word_tokenize(sent)
        wordsFiltered.extend([w for w in tokenized_sentence if w.lower() not in stopWords])

    # document = ' '.join(wordsFiltered)
    #
    # tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
    #
    # X_tfidf = tfidf_vectorizer.fit_transform([document])
    #
    # feature_names = tfidf_vectorizer.get_feature_names_out()
    # tfidf_values = X_tfidf.toarray()[0]
    #
    # return feature_names, tfidf_values
    return wordsFiltered

def calculate_difficulty(df):
    df = df.copy()

    word_counts = df['tokenized_chapter_names'].value_counts()

    df['count'] = df['tokenized_chapter_names'].map(word_counts)

    scaler = StandardScaler()
    df['z_scores'] = scaler.fit_transform(pd.DataFrame(df['count']))

    minmax_scaler = MinMaxScaler() #de 0 a 1
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

df['is_english'] = df['chapter'].apply(is_english)
df = df[df['is_english']]
df = df.drop(columns=['is_english', 'isbn'])

print(len(df))

df['author'] = df['author'].str.lower()
df['book_title'] = df['book_title'].str.lower()
df['category'] = df['category'].str.lower()
df['chapter'] = df['chapter'].str.lower()

# df[['tokenized_chapter_names', 'tokenized_chapter_values']] = df['chapter'].apply(lambda x: pd.Series(preprocessing(x)))
df['tokenized_chapter_names'] = df['chapter'].apply(lambda x: preprocessing(x))
# tokenized_chapter_names, tokenized_chapter_values = preprocessing(df['chapter'].srt.lower())
# df['tokenized_chapter_names'] = tokenized_chapter_names
# df['tokenized_chapter_values'] = tokenized_chapter_values

# df_exploded = df.explode(['tokenized_chapter_names', 'tokenized_chapter_values'])

# df_difficulty = calculate_difficulty(df_exploded)
tokenizer = AutoTokenizer.from_pretrained("RobPruzan/text-difficulty")
model = AutoModelForSequenceClassification.from_pretrained("RobPruzan/text-difficulty")

df_difficulty= df['tokenized_chapter_names'].apply(lambda x: model(**tokenizer(x, return_tensors="pt")))
df_difficulty_final = df.copy()
df_difficulty_final['difficulty'] = df_difficulty

# df_difficulty_final = df_difficulty.groupby(['author', 'book_title', 'category'])['difficulty'].mean().reset_index()
print(df_difficulty_final.info())
print(df_difficulty_final.isna().sum())
print(df_difficulty_final.describe())


df_difficulty_final['english_level'] = df_difficulty_final['difficulty'].apply(map_proficiency)

df_difficulty_final.to_csv('./output/difficulty_final2.csv')
