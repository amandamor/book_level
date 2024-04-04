import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from langdetect import detect
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


proficiency_intervals = {
    'A1': (0, 0.2),
    'A2': (0.2, 0.4),
    'B1': (0.4, 0.6),
    'B2': (0.6, 0.8),
    'C1': (0.8, 0.95),
    'C2': (0.95, 1.0)
}
stopWords = set(stopwords.words('english'))


def calculate_word_difficulty(tokenized_chapters, titles, authors, model, tokenizer):
    chapter_difficulty = {}

    progress_bar = tqdm(total=len(tokenized_chapters), desc="Processing")

    for title, author, words in zip(titles, authors, tokenized_chapters):
        tokenized_input = tokenizer(" ".join(words), return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            max_difficulty = torch.max(probabilities[:, 1]).item()

            key = (title, author)
            chapter_difficulty[key] = max_difficulty

        progress_bar.update(1)
    progress_bar.close()

    return chapter_difficulty

def dividir_em_sentencas(texto):
    sentencas = nltk.sent_tokenize(texto)
    return sentencas

def preprocessing(sentence):
    sentence = (''.join(char for char in sentence if not char.isdigit()))

    words = list(filter(lambda x: x not in stopWords, set(word_tokenize(sentence))))

    for punctuation in string.punctuation:
        words = list(map(lambda x: x.replace(punctuation, ''), words))
    words = list(map(lambda x: x.replace("'s", ''), words))

    wordsFiltered = words

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
        lang = detect(text[-100:])
        return lang == 'en'
    except:
        return False

def map_proficiency(difficulty):
    for proficiency, interval in proficiency_intervals.items():
        if interval[0] <= difficulty < interval[1]:
            return proficiency
    return None

df = pd.read_json('./output/final.json')

df['is_english'] = df['chapter'].apply(is_english)
df = df[df['is_english']]
df = df.drop(columns=['is_english', 'isbn'])

print(len(df))

df['author'] = df['author'].str.lower()
df['book_title'] = df['book_title'].str.lower()
df['category'] = df['category'].str.lower()
df['chapter'] = df['chapter'].str.lower()

# df[['tokenized_chapter_names', 'tokenized_chapter_values']] = df['chapter'].apply(lambda x: pd.Series(preprocessing(x)))
df['tokenized_chapter_words'] = df['chapter'].apply(lambda x: preprocessing(x))
# tokenized_chapter_names, tokenized_chapter_values = preprocessing(df['chapter'].srt.lower())
# df['tokenized_chapter_names'] = tokenized_chapter_names
# df['tokenized_chapter_values'] = tokenized_chapter_values

# df_exploded = df.explode(['tokenized_chapter_names', 'tokenized_chapter_values'])

# df_difficulty = calculate_difficulty(df_exploded)
# tokenizer = AutoTokenizer.from_pretrained("RobPruzan/text-difficulty")
# model = AutoModelForSequenceClassification.from_pretrained("RobPruzan/text-difficulty")
#
# df_difficulty= df['tokenized_chapter_words'].apply(lambda x: model(**tokenizer(" ".join(x), return_tensors="pt")))

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

difficulty_dict = calculate_word_difficulty(df['tokenized_chapter_words'],df['author'], df['book_title'], model, tokenizer)

df_difficulty = pd.DataFrame(list(difficulty_dict.items()), columns=['author_title', 'difficulty'])
df_difficulty[['author', 'title']] = pd.DataFrame(df_difficulty['author_title'].tolist(), index=df_difficulty.index)
df_difficulty.drop('author_title', axis=1, inplace=True)

df_difficulty['english_level'] = df_difficulty['difficulty'].apply(map_proficiency)

df_difficulty.to_csv('./output/difficulty_final_large_finetune.csv')
