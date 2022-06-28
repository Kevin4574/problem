from bs4 import BeautifulSoup
import requests


url = 'https://www.udacity.com/courses/all'
r = requests.get(url)
soup = BeautifulSoup(r.text,
                     'html5lib')
summary = soup.find_all('li',class_ = 'card_container__25Drk')
print(summary)





