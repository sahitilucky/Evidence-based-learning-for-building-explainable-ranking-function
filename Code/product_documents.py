import io
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import math
import random
from sklearn.decomposition import LatentDirichletAllocation
import io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import pickle
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import time
stopwords = stopwords.words()
lemmatizer = WordNetLemmatizer()
def get_string(value):
    strVal = value
    try:
        strVal = str(value)
    except:
        strVal = value.encode('utf-8')
    return strVal

def preprocess(text, stopwords, lemmatizer):
    text = text.replace('\n',' ').replace('\r','')
    #text = get_string(text)
    if isinstance(text, str):
        try:
            text = unicode(text, 'utf-8')
        except:
            text = ''
    
    #print text
    chars = '@#$%&*()!^:;?<>/[]"~.,\'=+|-\\{\\}_`'
    '''
    for c in chars:
        text = text.replace(c,' ')
    '''
    begin = time.time()
    words = word_tokenize(text.lower())
    end = time.time()
    #print ('tokenizing:', (end-begin))
    #print words
    #words = [get_string(w) for w in words]
    '''
    begin = time.time()
    words = filter(lambda l : l not in stopwords, words)
    end = time.time()
    print ('Stopwords pruning:', (end-begin))
    '''
    '''
    new_words = []
    for word in words:
        if any((c in word) for c in chars):
            continue:
        else:
            new_words += [word]
    '''
    begin = time.time()
    words = filter(lambda l: not any((c in l) for c in chars), words)
    end = time.time()
    #print ('Chars pruning:', (end-begin))
    #print words
    lemmas = []
    for word in words:
        if isinstance(word, str):
            try:
                word = unicode(word, 'utf-8')
            except:
                word = ''
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    text = ' '.join(lemmas)
    return text
'''
Product_documents = {}
docid = None
idx = 0
with open('../../Web_data/document_corpus_sorted.txt', 'r') as infile:
    for line in infile:
        if '<seperator>' in line:
            idx+=1
            if docid is not None:
                #print docid
                text = preprocess(text)
                Product_documents[docid] = text
                #print (text)
            docid = line.split('<seperator>')[0]
            text = line.split('<seperator>')[1]
            #print 'coming here'
        else:
            text += line.strip()
        if idx%500==0:
            print idx
print len(Product_documents.keys())

pickle.dump(Product_documents, open("../../Web_data/Product_documents_corpus_sorted_2.p","wb" ))
tf_vectorizer = CountVectorizer(lowercase = True)
Product_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.fit_transform(Product_documents.values())
pickle.dump( tf_vectorizer, open("../../Web_data/tf_vectorizer_corpus_sorted_2.p","wb" ))
tf_feature_names = tf_vectorizer.get_feature_names()
pickle.dump( tf_feature_names, open("../../Web_data/tf_feature_names_corpus_sorted_2.p","wb" ) )
print len(tf_vectorizer.get_feature_names())
'''
