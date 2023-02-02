import io
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import math
import random
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import pickle
import time
import numpy as np
import string
from sklearn.preprocessing import MinMaxScaler
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
stopwords = stopwords.words()
stopwords = dict(zip(stopwords, range(len(stopwords))))
lemmatizer = WordNetLemmatizer()
lemmatized_words = {}
#table = {char: None for char in string.punctuation}
#table = string.maketrans("", "", string.punctuation)
regex = re.compile('[%s]' % re.escape(string.punctuation))
def preprocess(text):
    #print (text)
    text = regex.sub('', text)
    words = text.lower().split()
    #print (words)
    #lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        try:
            s_word = stopwords[word]
            #print ('came here omg')
        except:
            if isinstance(word, str):
                try:
                    word = unicode(word, 'utf-8')
                except:
                    word = ''
            #print (word)
            #print ('came here amayya')
            try:
                lemmas.append(lemmatized_words[word])
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemmas.append(lemma)
                lemmatized_words[word] = lemma
    #print (lemmas)
    text = ' '.join(lemmas)
    #print (text)
    return text