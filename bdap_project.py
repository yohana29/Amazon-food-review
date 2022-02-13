# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:32:25 2021

@author: yohan
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from wordcloud import WordCloud


def model_fitfunction(independent_variable, dependent_variable, model,clf_model,coef_show=1):
    
    X_c = model.fit_transform(independent_variable)
    print('# features: {}'.format(X_c.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X_c, dependent_variable, random_state=0)
    print('# train records: {}'.format(X_train.shape[0]))
    print('# test records: {}'.format(X_test.shape[0]))
    clf = clf_model.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print ('Model Accuracy: {}'.format(acc))
    
reviews = pd.read_csv('Reviews.csv')
reviews.dropna(inplace=True)

# checking if there is any null value
reviews.isnull().sum()
reviews.head()
reviews.columns

independent_variable=reviews['Text'] 
dependent_variable=reviews['Score']

c = CountVectorizer(stop_words = 'english')
model_fitfunction(independent_variable, dependent_variable, c, LogisticRegression())


tfidf = TfidfVectorizer(stop_words = 'english')
model_fitfunction(independent_variable, dependent_variable, tfidf, LogisticRegression())

tfidf_n = TfidfVectorizer(ngram_range=(1,2),stop_words = 'english')
model_fitfunction(independent_variable, dependent_variable, tfidf_n, LogisticRegression())