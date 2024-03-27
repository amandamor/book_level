import pandas as pd
import requests
from lxml import html
import os

os.makedirs('./output', exist_ok=True)

url = "https://www.goodreads.com/list/show/183940.Best_Books_of_2023"

data_request = requests.get(url=url)
data_request = html.fromstring(data_request.content)

linhas = data_request.xpath(r'//td[a[@class="bookTitle"]]')

lista_livros = []

for linha in linhas:
    book_title = linha.xpath(r"./a[@class='bookTitle']/span")[0].text
    author_title = linha.xpath(r"./span/div/a[contains(@class, 'authorName')]/span")[0].text

    lista_livros.append({'book_title': book_title, 'author': author_title})

dataframe = pd.DataFrame(lista_livros)

dataframe.to_json('./output/books.json', orient='records')