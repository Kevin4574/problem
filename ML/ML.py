# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import re
import time
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

path = r'D:\金融股票\Udacity\Data Science\3. Data Engineering\5. Project - Disaster Response Pipeline\data'

# load data from database
engine = create_engine('sqlite:///' + path + '\ETL_Cleaned.db')
df = pd.read_sql_table('message',engine)

# replace all the 2 in related column with 1
df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

feature = df['message']
label = df.iloc[:,4:]

# define a function to tokenize and clean the feature
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

vectorizer = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
svm = MultiOutputClassifier(SVC())

# split to train test data
feature_train, feature_test, label_train, label_test = train_test_split(feature,
                                                                        label,
                                                                        test_size=0.9,
                                                                        random_state=42)


count = vectorizer.fit_transform(feature_train)
idf = tfidf.fit_transform(count)

svm.fit(idf,label_train)

