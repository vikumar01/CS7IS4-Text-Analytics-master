# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 11:10:50 2020

@author: mankadp
"""
import os,re
import pandas as pd
import readability
import numpy as np
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import pytextrank

def counting_features(texts):
    pronounCount = np.zeros(len(texts)).reshape(-1,1)
    prepositionCount = np.zeros(len(texts)).reshape(-1,
    conjunctionCount = np.zeros(len(texts)).reshape(-1,1)

    complexWordsCount = np.zeros(len(texts)).reshape(-1,1)
    longWordsCount = np.zeros(len(texts)).reshape(-1,1)
    syllablesCount = np.zeros(len(texts)).reshape(-1,1)

    ## type token ratio is the no of unique words divided by total words
    typeTokenRatio = np.zeros(len(texts)).reshape(-1,1)
    wordCount = np.zeros(len(texts)).reshape(-1,1)


    for i,text in enumerate(texts):
        # print(text)
        score = readability.getmeasures(text,lang="en")
        sentenceInfo = score["sentence info"]
        wordUsage = score["word usage"]
        # word usages
        pronounCount[i] = wordUsage['pronoun']
        prepositionCount[i] = wordUsage['preposition']
        conjunctionCount[i] = wordUsage['conjunction']
        # sentence info
        complexWordsCount[i] = sentenceInfo['complex_words']
        longWordsCount[i] = sentenceInfo['long_words']
        syllablesCount[i] = sentenceInfo['syllables']
        typeTokenRatio[i] = sentenceInfo['type_token_ratio']
        wordCount[i] = sentenceInfo['words']

    ## Combining all of them into one
    featureCounts = pd.DataFrame(data = np.concatenate((pronounCount,prepositionCount,conjunctionCount,complexWordsCount
                                           ,longWordsCount,syllablesCount,typeTokenRatio,wordCount),axis=1),
    columns=["pronounCount","prepositionCount","conjunctionCount","complexWordsCount","longWordsCount",
             "syllablesCount","typeTokenRatio","wordCount"])
    return featureCounts


def preprocessing(texts):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    cleanedText = []
    entity = []

    for i,text in enumerate(texts):
        words = ""
        temp_entity = ""
        ## Lemmatization
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

        ## removes Special Characters
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern,"",str(text))
        # print(text)

        ## seperates each word and lowercases as well
        tokens = tokenizer.tokenize(text.lower())
        ## removes spaces in each words, if any
        tokens = [token.strip() for token in tokens]
        ## filters stop words in tokens
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        for word in filtered_tokens:
            words += word + " "
        cleanedText.append(words)

        ## Named entity recognition
        words = nlp(words)
        temp_entity += str([word for word in words if word.ent_type])
        entity.append(temp_entity)

        # print(words)
        # print(cleanedText)
        # break
        if i%500 == 0:
            print("Steps Done:",i)

    return cleanedText,entity

def main(input_file,output_file):

    # the number of features
    STEPS = 100
    data = pd.read_csv(input_file).iloc[:,:]

    try:
        pass
        os.system("pip install -r requirements.txt")
        nltk.download('stopwords')
        os.system("python -m spacy download en_core_web_sm")
    except:
        pass
    '''
    '''

    # feature = data.iloc[:,1]
    texts = data['text']
    authors = data['author']

# =============================================================================
    ## Consists of all the counted features
    featureCounts=counting_features(texts)
    print("Feature Counts DONE!")
# =============================================================================


# =============================================================================
    ## Lemmatization, Removing special characters, Stop words Removal
    cleanedText,entity = preprocessing(texts)
    print("Text Cleaning DONE!")
    featureCounts["cleaned"] = cleanedText
    featureCounts["entity"] = entity
    featureCounts["author"] = data["author"]
    try:
        featureCounts.to_csv(output_file,index=False)
    except:
        print("close your file for system to update")
# =============================================================================

if __name__ == "__main__":

    if not os.path.exists("Rajdhaani.csv"):
        main("train.csv","Rajdhaani.csv")
        print("Hey BRO, Running for the first time")

# =============================================================================
    # Training the model / Classifier

    data = pd.read_csv("Rajdhaani.csv").fillna("undefined")
    labels = data.iloc[:,-1]
    train = data.iloc[:,:9]

    # Encoding labels
    from sklearn.preprocessing import OrdinalEncoder
    labels = OrdinalEncoder().fit_transform(pd.DataFrame(labels).to_numpy().reshape(-1,1))

    ## train test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(train, labels, test_size=0.2,shuffle = True)

    ## TFIDF vectoriser
    from sklearn.feature_extraction.text import TfidfVectorizer
    X_train_cleaned = pd.DataFrame(TfidfVectorizer().fit_transform(X_train["cleaned"]).toarray())
    X_test_cleaned = pd.DataFrame(TfidfVectorizer().fit_transform(X_test["cleaned"]).toarray())
    print("Vectorisation Done")

    ## Concatenation
    X_train = pd.concat([X_train.iloc[:,:8],X_train_cleaned])
    X_test = pd.concat([X_test.iloc[:,:8],X_test_cleaned])

    ## model selection
    import lightgbm
    X_train = lightgbm.Dataset(X_train, label=Y_train)
    X_test = lightgbm.Dataset(X_test, label=Y_test)

    params = {
        'task': 'train',
        'obective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'verbose': 1
        }
    model = lightgbm.train(params,
                       X_train,
                       valid_sets=[X_train],
                       verbose_eval=10,
                       num_boost_round=1000,
                       early_stopping_rounds=100)

# =============================================================================
    # Prediction
    Y_predict = model.predict(Y_test)

    from sklearn.metrics import mean_squared_error,confusion_matrix
    confMatrix = confusion_matrix(Y_test,Y_predict)

    score = np.sqrt(mean_squared_error(Y_test,Y_predict))

    print("\n Final Score is",score)


# =============================================================================
    # Testing the model

# =============================================================================










