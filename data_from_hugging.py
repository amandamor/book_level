import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression

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


df_train = pd.read_parquet('./output/train.parquet')
df_test = pd.read_parquet('./output/test.parquet')
df_val = pd.read_parquet('./output/validation.parquet')

df_concat = pd.concat([df_train, df_test, df_val], axis=0)

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: x.encode('latin1', errors='ignore').decode('utf-8'))
df_concat['book_id'] = df_concat['book_id'].str.lower()
df_concat['book_id'] = df_concat['book_id'].str.replace(",", "")
df_concat['book_id'] = df_concat['book_id'].str.replace(":", "")
df_concat['book_id'] = df_concat['book_id'].str.replace("\\", "")
df_concat['chapter'] = df_concat['chapter'].str.replace(",", "")
df_concat['chapter'] = df_concat['chapter'].str.replace("\\", "")

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: re.search(r'^(.*?)\.', x).group(1) if re.search(r'^(.*?)\.', x) else x)
df_concat = df_concat.drop_duplicates(subset=['book_id'], keep='first')
df_concat = df_concat[['book_id', 'chapter']]
df_concat[['tokenized_chapter_names', 'tokenized_chapter_values']] = df_concat['chapter'].apply(lambda x: pd.Series(preprocessing(x)))

df_exploded = df_concat.explode(['tokenized_chapter_names', 'tokenized_chapter_values'])

df_difficulty = calculate_difficulty(df_exploded)

# print(df_difficulty.isna().sum())
# print(df_difficulty.describe())
# print(df_difficulty.info())

################### MODEL ###################

X = df_difficulty[['tokenized_chapter_values']]
y = df_difficulty['difficulty']    # Variável alvo (dificuldade)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    # PolynomialFeatures(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    # SVR(),
    MLPRegressor()
]

model_mse = []

for model in models:
    print(model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    model_mse.append({'model': model, 'mse': mse})

print('\n'.join([f"{result['model']}: {result['mse']}" for result in model_mse]))


""" 
MLPRegressor()
LinearRegression(): 1.3132227493882662e-26
Ridge(): 1.0205158791662882e-25
Lasso(): 1.3927275455519268e-06
DecisionTreeRegressor(): 2.103404000039251e-29
RandomForestRegressor(): 6.39943491536164e-30
GradientBoostingRegressor(): 4.744948432629651e-07
SVR(): 0.007276583661479124
MLPRegressor(): 8.959357549481053e-05
"""

"""
New_model with values per word
LinearRegression(): 0.04692219800103661
Ridge(): 0.046929850922370925
Lasso(): 0.05437747109587966
DecisionTreeRegressor(): 0.03932792662428544
RandomForestRegressor(): 0.03880273437033058
GradientBoostingRegressor(): 0.04044777722332121
MLPRegressor(): 0.04129191294145817
"""

