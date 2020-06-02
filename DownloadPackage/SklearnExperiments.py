# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:10:14 2020

@author: Machachane
"""

print('\n1 -------------------------------------------------------------------\n')

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
print(list(newsgroups_train.target_names))

print(newsgroups_train.filenames.shape)
print(newsgroups_train.target.shape)
print(newsgroups_train.target[:10])


print('\n2 -------------------------------------------------------------------\n')

cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)

print(list(newsgroups_train.target_names))
print(newsgroups_train.filenames.shape)

print(newsgroups_train.target.shape)

print(newsgroups_train.target[:10])


print('\n3 -------------------------------------------------------------------\n')

from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

print('\n4 -------------------------------------------------------------------\n')

print(vectors.nnz / float(vectors.shape[0]))

print('\n5 -------------------------------------------------------------------\n')

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

print('\n6 -------------------------------------------------------------------\n')

import numpy as np
def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf, vectorizer, newsgroups_train.target_names)

print('\n7 -------------------------------------------------------------------\n')

newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))

print('\n8 -------------------------------------------------------------------\n')

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

"""
https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html

"""
