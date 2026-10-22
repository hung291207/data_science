#REPORT LINK: https://docs.google.com/document/d/1JYkjsB89HRxMTgCKSd3PbliDRnZazmxJKiScpBBCrw0/edit?tab=t.0

import requests
import re
from bs4 import BeautifulSoup as bs
import pandas as pd
from urllib.parse import urljoin
import os

# Create directories if they don't exist
os.makedirs('data_sheet', exist_ok=True)
os.makedirs('images', exist_ok=True)

#stores book data
books = []

LINK = "https://books.toscrape.com/"

r = requests.get(LINK)
r.encoding = "utf-8"

soup = bs(r.text, 'html.parser')

category_list = soup.find_all('a')

#scrapes a book's details
def scrape_book(book, page_url, category):
    title = book.find('h3').find('a')['title']
    price = float(book.find('p', class_='price_color').string[2:])
    availability = book.find('p', class_='instock availability').text.strip()
    category = category
    rating = book.find('p', class_='star-rating')['class'][1]
    link = urljoin(page_url, book.find('h3').find('a')['href'])
    thumbnail_url = 'https://books.toscrape.com/' + book.find('img', class_='thumbnail')['src']
    thumbnail_file_name = save_thumbnail_image(thumbnail_url, title)


    #writes into a dictionary & add to 'books'
    book_data = {
        'Title': title,
        'Price': price,
        'Availability': availability,
        'Category': category,
        'Rating': rating,
        'Link': link,
        'Thumbnail URL': thumbnail_url,
        'Thumbnail File Name': thumbnail_file_name
    }

    books.append(book_data)

    #prints out results
    print(title)
    print(price)
    print(availability)
    print(category)
    print(rating)
    print(link)
    print(thumbnail_url)

#downloads the thumbnail image
def save_thumbnail_image(url, title):
    response = requests.get(url)
    sanitized_title = "".join([c if c.isalnum() else "_" for c in title])
    image_path = os.path.join('images', f"{sanitized_title}.jpg")
    with open(image_path, 'wb') as file:
        file.write(response.content)
    return image_path


#scrapes from a certain category
def scrape_category(name):

    #loops through category list for the searched category
    for link in category_list:
        if link.get_text(strip=True) == name:
            category_url = LINK + link.get('href')

            #loops through the pages
            page = 1
            while True:
                #replaces the category url with the page url
                url = category_url.replace("index.html", f"page-{page}.html")

                r = requests.get(url)

                #if the url returns a 404 error -> stop (out of pages)
                if r.status_code == 404:
                    break

                #get soup of the category
                category_soup = bs(r.text, 'html.parser')
                #find all objects in 'article class=product pod' aka books
                category_books = category_soup.find_all('article', class_= 'product_pod')

                for book in category_books:
                    scrape_book(book, url, name)

                page += 1


scrape_category('Fiction')
scrape_category('Nonfiction')

# Save the data to a CSV file
df = pd.DataFrame(books)
df.to_csv('data_sheet/books_data.csv', index=False)
print("Data saved to data_sheet/books_data.csv")






