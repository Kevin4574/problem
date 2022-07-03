import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from encodings.aliases import aliases
import chardet
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from custom_transformer import StartingVerbExtractor
from numpy import hstack

np.random.seed(42)

path = r'D:\金融股票\Udacity\Data Science\3. Data Engineering\4. Machine Learning Pipelines\data'
file = path + '\corporate_messaging.csv'

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data():
    df = pd.read_csv(file, encoding='latin-1')
    df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
    X = df.text.values
    y = df.category.values
    return X, y

# create tokenize function that can take [text] as input and output[token,token,token,....]
def tokenize(text):
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

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

def main():
    # load data and perform train text split
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # instantiate transformers and classifiers
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    verb_vectorizer = StartingVerbExtractor()
    clf = RandomForestClassifier()

    # fit and transform the training data
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    print(len(X_train_tfidf.toarray()))

    # fit transform for start verb
    star_verb = verb_vectorizer.fit_transform(X_train)
    print(len(star_verb))

    # add the new feature
    fea = hstack([X_train_tfidf,star_verb])
    print(fea)
    exit()

    # train classifier
    clf.fit(X_train_tfidf, y_train)

    # transform (no fitting) the test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    # predict on test data
    y_pred = clf.predict(X_test_tfidf)

    # display results
    display_results(y_test, y_pred)


main()










