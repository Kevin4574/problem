import pandas as pd
import numpy as np
import requests
import joblib
from urllib.request import urlopen
import json
import ast
from bs4 import BeautifulSoup
from selenium import webdriver

url = 'https://canelink.miami.edu/psc/UMIACPRD/EMPLOYEE/SA/s/WEBLIB_HCX_CM.H_COURSE_CATALOG.FieldFormula.IScript_CatalogCourseDetails?institution=MIAMI&course_id=126805&crse_offer_nbr=1&crs_topic_id=0&acad_career='
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.62',
           'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
           }

# content = requests.get(url,headers=headers).content.decode('utf-8')
# content = json.loads(content)
# print(content)
# course_detail = content['course_details']
# df = pd.DataFrame(course_detail.items())


url2 = 'https://canelink.miami.edu/psc/UMIACPRD/EMPLOYEE/SA/s/WEBLIB_HCX_RE.H_DEGREE_PROGRESS.FieldFormula.IScript_DegreeProgress?acad_career=GRAD&institution=MIAMI'
content = requests.get(url2,headers=headers).content.decode('utf-8')
print(content)
