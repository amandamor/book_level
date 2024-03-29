import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


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

    return wordsFiltered

def calculate_difficulty(df):
    df = df.copy()

    # Calcular o número de palavras únicas
    word_counts = df['tokenized_chapter'].value_counts()

    # Mapeando os valores de contagem de volta ao DataFrame original
    df['count'] = df['tokenized_chapter'].map(word_counts)
    # repeated_tokenized_chapter_count = [tokenized_chapter_count] * len(df)

    # Escalonar o número de palavras únicas
    scaler = StandardScaler()
    df['z_scores'] = scaler.fit_transform(pd.DataFrame(df['count']))

    # Escalonar os z-scores para o intervalo [0, 1]
    minmax_scaler = MinMaxScaler()
    df['difficulty'] = minmax_scaler.fit_transform(df['z_scores'].values.reshape(-1, 1))

    return df



df_train = pd.read_parquet('output/train.parquet')
df_test = pd.read_parquet('output/test.parquet')
df_val = pd.read_parquet('output/validation.parquet')

df_concat = pd.concat([df_train, df_test, df_val], axis=0)

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: x.encode('latin1', errors='ignore').decode('utf-8'))
df_concat['book_id'] = df_concat['book_id'].str.lower()
df_concat['book_id'] = df_concat['book_id'].str.replace(",", "")
df_concat['book_id'] = df_concat['book_id'].str.replace(":", "")
df_concat['book_id'] = df_concat['book_id'].str.replace("\\", "")

df_concat['book_id'] = df_concat['book_id'].apply(lambda x: re.search(r'^(.*?)\.', x).group(1) if re.search(r'^(.*?)\.', x) else x)
df_concat = df_concat.drop_duplicates(subset=['book_id'], keep='first')
df_concat = df_concat[['book_id', 'chapter']]
df_concat['tokenized_chapter'] = df_concat['chapter'].apply(preprocessing)

df_livros_exploded = df_concat.explode('tokenized_chapter')

df_difficulty = calculate_difficulty(df_livros_exploded)

# print(df_difficulty.isna().sum())
# print(df_difficulty.describe())
# print(df_difficulty.info())
################### MODEL ###################

X = df_difficulty[['count']]  # , 'tokenized_chapter'
y = df_difficulty['difficulty']     # Variável alvo (dificuldade)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = [
    # PolynomialFeatures(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    MLPRegressor()
]

model_mse = []

for model in models:
    print(model)
    model.fit(X_train, y_train)

    # Prevendo a dificuldade das palavras no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliando a precisão do modelo
    mse = mean_squared_error(y_test, y_pred)
    model_mse.append({'model': model, 'mse': mse})

print('\n'.join([f"{result['model']}: {result['mse']}" for result in model_mse]))


a = 5
