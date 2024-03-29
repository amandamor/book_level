import pandas as pd
import requests
from lxml import html
import os
import re

os.makedirs('./output', exist_ok=True)

url_list = ["https://www.goodreads.com/list/show/143500.Best_Books_of_the_Decade_2020_s",
            "https://www.goodreads.com/list/best_of_year/2024?id=196307.Best_Books_of_2024",
            "https://www.goodreads.com/list/best_of_year/2022?id=171064.Best_Books_of_2022",
            "https://www.goodreads.com/list/best_of_year/2021?id=157516.Best_Books_of_2021",
            "https://www.goodreads.com/list/best_of_year/2020?id=143444.Best_Books_of_2020",
            "https://www.goodreads.com/list/show/114787.Best_Books_2019",
            "https://www.goodreads.com/list/show/119307.Best_Books_of_2018",
            "https://www.goodreads.com/list/best_of_year/2017?id=107026.Best_Books_of_2017",
            "https://www.goodreads.com/list/best_of_year/2016?id=95160.Best_Books_of_2016",
            "https://www.goodreads.com/list/best_of_year/2015?id=86673.Best_Books_of_2015",
            "https://www.goodreads.com/list/best_of_year/2014?id=47649.Best_Books_of_2014",
            "https://www.goodreads.com/list/best_of_year/2013?id=27345.Best_Books_Published_in_2013",
            "https://www.goodreads.com/list/best_of_year/2012?id=15604.Best_Books_of_2012",
            "https://www.goodreads.com/list/best_of_year/2011?id=8226.Best_Books_of_2011",
            "https://www.goodreads.com/list/best_of_year/2010?id=3957.Best_Books_of_2010",
            "https://www.goodreads.com/list/best_of_year/2009?id=1297.Best_Books_of_2009",
            "https://www.goodreads.com/list/best_of_year/2008?id=4.Best_Books_of_2008",
            "https://www.goodreads.com/list/best_of_year/2007?id=29.Best_Books_of_2007",
            "https://www.goodreads.com/list/best_of_year/2006?id=20.Best_Books_of_2006",
            "https://www.goodreads.com/list/best_of_year/2005?id=35.Best_Books_of_2005",
            "https://www.goodreads.com/list/best_of_year/2004?id=75.Best_Books_of_2004",
            "https://www.goodreads.com/list/best_of_year/2003?id=76.Best_Books_of_2003",
            "https://www.goodreads.com/list/best_of_year/2002?id=77.Best_Books_of_2002",
            "https://www.goodreads.com/list/best_of_year/2001?id=54.Best_Books_of_2001",
            "https://www.goodreads.com/list/best_of_year/2000?id=78.Best_Books_of_2000",
            "https://www.goodreads.com/list/show/35080",
            "https://www.goodreads.com/list/show/35177",
            "https://www.goodreads.com/list/show/117146",
            "https://www.goodreads.com/list/show/141019",
            "https://www.goodreads.com/list/show/141027",
            "https://www.goodreads.com/list/show/141033",
            "https://www.goodreads.com/list/show/7",
            "https://www.goodreads.com/list/show/4893",
            "https://www.goodreads.com/list/show/21995",
            "https://www.goodreads.com/list/show/3810.Best_Cozy_Mystery_Series",
            "https://www.goodreads.com/list/show/179830",
            "https://www.goodreads.com/list/show/79359.Best_Mysteries_from_the_2000s",
            "https://www.goodreads.com/list/show/914",
            "https://www.goodreads.com/list/show/79879",
            "https://www.goodreads.com/list/show/73283",
            "https://www.goodreads.com/list/show/1.Best_Books_Ever",
            "https://www.goodreads.com/list/show/47.Best_Dystopian_and_Post_Apocalyptic_Fiction",
            "https://www.goodreads.com/list/show/300",
            "https://www.goodreads.com/list/show/51",
            "https://www.goodreads.com/list/show/10762.Best_Book_Boyfriends",
            "https://www.goodreads.com/list/show/397.Best_Paranormal_Romance_Series",
            "https://www.goodreads.com/list/show/12362.All_Time_Favorite_Romance_Novels",
            "https://www.goodreads.com/list/show/12066.College_Romance",
            "https://www.goodreads.com/list/show/57.Best_Ever_Contemporary_Romance_Books",
            "https://www.goodreads.com/list/show/26495.Best_Woman_Authored_Books",
            "https://www.goodreads.com/list/show/8329.Best_Romance_Books_Ever",
            "https://www.goodreads.com/list/show/97747.Slow_Burn_Romance",
            "https://www.goodreads.com/list/show/10942.Our_Favorite_Indie_Reads",
            "https://www.goodreads.com/list/show/83916.Historical_Fiction_2016",
            "https://www.goodreads.com/list/show/559.What_To_Read_After_Harry_Potter",
            "https://www.goodreads.com/list/show/1023.Best_Strong_Female_Fantasy_Novels",
            "https://www.goodreads.com/list/show/312.Best_Humorous_Books",
            "https://www.goodreads.com/list/show/692.Best_Science_Books_Non_Fiction_Only",
            "https://www.goodreads.com/list/show/691.Best_Self_Help_Books",
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