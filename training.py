# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:53:10 2020

@author: Piyush
"""

import pandas as pd
import matplotlib.pyplot as plt

def textfeatures(features):
	phrase=[]
	for feature in list(features):
		words = ""
# 		print("feature",feature,type(feature))
		for word in feature.split(","):
			word = word.replace(" ","")
			word = word.replace("'","")
			word = word.replace("[","")
			word = word.replace("]","")
			words += word + " "
# 			print("word",word)
		phrase.append(words)
# 		break
	return phrase

bigstep = 19580
data = pd.read_csv(r"D:\Intelligent Systems\TextAnalytics\Authorship-Attribution\final_features.csv").iloc[:bigstep,:].fillna("undefined")
labels = data["author"]

## text features
nouns = data["nouns"]
named_ents = data["named_ents"]
verbs = data["verb"]
adjs = data["adj"]
noun_phrases = data["noun_phrases"]

## count features
count_nouns = data["noun_tokens"]
count_named_ents = data["entity_tokens"]
count_verbs = data["verb_tokens"]
count_adjs = data["adj_tokens"]
count_noun_phrases = data["noun_phrases_tokens"]

## extra count features
unique_words = data["num_unique_words"]
stopwords = data["num_stopwords"]
articles = data["article"]
punc = data["num_punctuations"]
mean_word_len = data["mean_word_len"]
non_voc = data["n_non_voc"]

noun_phrases = textfeatures(noun_phrases)
nouns = textfeatures(nouns)
verbs = textfeatures(verbs)
named_ents = textfeatures(named_ents)
adjs = textfeatures(adjs)


# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
nouns = pd.DataFrame(tfidf.fit_transform(nouns).toarray())
# verbs = pd.DataFrame(tfidf.fit_transform(verbs).toarray())
# named_ents = pd.DataFrame(tfidf.fit_transform(named_ents).toarray())
# adjs = pd.DataFrame(tfidf.fit_transform(adjs).toarray())
# noun_phrases = pd.DataFrame(tfidf.fit_transform(noun_phrases).toarray())

## ordinal encoder
from sklearn.preprocessing import OrdinalEncoder
labels = pd.DataFrame(OrdinalEncoder().fit_transform(labels.to_numpy().reshape(-1,1)))
print("encoded")

## joining
X = pd.concat([count_nouns,count_named_ents,count_verbs,count_adjs,count_noun_phrases,
              unique_words,stopwords,articles,punc,mean_word_len,non_voc,],axis=1)
print("joined",X.shape)

## de allocating memory
# nouns = named_ents = verbs = adjs = count_nouns = count_named_ents = count_verbs = count_adjs = count_noun_phrases = unique_words = stopwords = articles = punc = mean_word_len = non_voc = 0
print("de allocation")

## train test split
from sklearn.model_selection import train_test_split
steps = 15000
X_train,X_test,Y_train,Y_test = train_test_split(X.iloc[:steps,:], labels.iloc[:steps,:], test_size=0.1,shuffle = True)
print("split")

# training a model
# from sklearn.svm import SVC
# clf = SVC(kernel = "linear",C=0.025, random_state = 13)
# clf.fit(X_train,Y_train)
# y_pred = clf.predict(X_test)

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=42)
# clf.fit(X_train,Y_train)
# Y_predict = clf.predict(X_test)

import lightgbm as lgb
mdl = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=63, max_depth=-1, learning_rate=0.1, n_estimators=10000)
mdl.fit(X_train, Y_train.values.ravel())
Y_predict = mdl.predict(X_test)


## Validation set metrics
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,precision_recall_fscore_support
confMatrix = confusion_matrix(Y_test,Y_predict)
accuracy_noun = accuracy_score(Y_test,Y_predict)
precision_recall_fscore = precision_recall_fscore_support(Y_test,Y_predict)
print("[count features]\n Accurracy {} \nPrecision recall {} \n confusion matrix \n{} ".format(accuracy_noun,precision_recall_fscore,confMatrix))

## plotting feature relevance
fig, ax = plt.subplots(figsize=(12,12))
lgb.plot_importance(mdl,max_num_features=15,height=0.5,ax=ax)
plt.show()

## testing set metrics
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,precision_recall_fscore_support
Y_test_predict = clf.predict(X.iloc[steps:bigstep-1,:])
Y_test_final = labels.iloc[steps:bigstep-1,:]

confMatrix = confusion_matrix(Y_test_final,Y_test_predict)
accuracy_noun = accuracy_score(Y_test_final,Y_test_predict)
precision_recall_fscore = precision_recall_fscore_support(Y_test_final,Y_test_predict)
print("[count features][TEST]\n Accurracy {} \nPrecision recall {} \nconfusion matrix \n{} ".format(accuracy_noun,precision_recall_fscore,confMatrix))
