import pandas as pd
import requests
from lxml import html
import os
import re

os.makedirs('./output', exist_ok=True)

url_list = ["https://www.goodreads.com/list/show/143500.Best_Books_of_the_Decade_2020_s",
            "https://www.goodreads.com/list/show/35080",
            "https://www.goodreads.com/list/show/35177",
            "https://www.goodreads.com/list/show/117146",
            "https://www.goodreads.com/list/show/141019",
            "https://www.goodreads.com/list/show/141027",
            "https://www.goodreads.com/list/show/141033",
            "https://www.goodreads.com/list/show/7",
            "https://www.goodreads.com/list/show/4893",
            "https://www.goodreads.com/list/show/21995",
            "https://www.goodreads.com/list/show/91.Best_Historical_Fiction_of_the_21st_Century",
            "https://www.goodreads.com/list/show/26068",
            "https://www.goodreads.com/list/show/155086.Popular_Kindle_Notes_Highlights_on_Goodreads",
            "https://www.goodreads.com/list/show/183940.Best_Books_of_2023",
            "https://www.goodreads.com/list/show/3810.Best_Cozy_Mystery_Series",
            "https://www.goodreads.com/list/show/1043.Books_That_Should_Be_Made_Into_Movies"]

list_books = []

for url in url_list:
    print("Downloading: " + url)

    data_request = requests.get(url=url)
    data_request = html.fromstring(data_request.content)

    lines = data_request.xpath(r'//td[a[@class="bookTitle"]]')


    for line in lines:
        book_title = line.xpath(r"./a[@class='bookTitle']/span")[0].text
        author_title = line.xpath(r"./span/div/a[contains(@class, 'authorName')]/span")[0].text

        list_books.append({'book_title': book_title, 'author': author_title})

dataframe = pd.DataFrame(list_books)
dataframe['book_title'] = dataframe['book_title'].apply(lambda x: x.encode('latin1', errors='ignore').decode('utf-8'))
dataframe['author'] = dataframe['author'].apply(lambda x: x.encode('latin1', errors='ignore').decode('utf-8'))
dataframe['book_title'] = dataframe['book_title'].apply(lambda x: x.replace("\u200b", ""))
dataframe['book_title'] = dataframe['book_title'].apply(lambda x: x.replace('"', ""))

dataframe['book_title'] = dataframe['book_title'].apply(lambda x: re.search(r'^(.*?)\s*\(', x).group(1) if re.search(r'^(.*?)\s*\(', x) else x)

dataframe = dataframe.drop_duplicates(subset=['book_title', 'author'], keep='first')
dataframe = dataframe.sort_values(by=['author'])

dataframe.to_csv('./output/books.csv', index=False)
dataframe.to_json('./output/books.json', orient='records')