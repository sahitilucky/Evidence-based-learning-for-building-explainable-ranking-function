
from product_documents import *
import pickle
import json
import os
import time
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
stopwords = stopwords.words()
lemmatizer = WordNetLemmatizer()
def get_string(value):
    strVal = value
    try:
        strVal = str(value)
    except:
        strVal = value.encode('utf-8')
    return strVal

doc_no = ''
document_corpus = {}
corpus_folder = '/shared/trec2/shared/shared-data/gov2-data/gov2-corpus/unbxd/'
files = os.listdir(corpus_folder)
files = filter(lambda l: (('.txt' in l) and ('_anchor.txt' not in l)) and (l!='documents_list.txt'), files)
print files
for file in files:
	i = 1
	with open(corpus_folder + file, 'r') as infile:
	    for line in infile:
	        line = get_string(line)
	        if i%4 == 1:
	            doc_no = line.strip().split('docno:')[1]
	            document_corpus[doc_no] = {}
	            #print (doc_no)
	        if i%4 == 2:
	            document_corpus[doc_no]['title'] = line.strip().split('title:')[1]
	            #print (line.strip().split('title:')[1])
	        if i%4 == 3:
	            document_corpus[doc_no]['url'] = line.strip().split('url:')[1]
	            #print (line.strip().split('url:')[1])
	        if i%4 == 0:
	            document_corpus[doc_no]['body'] = line.strip().split('body:')[1]
	        i += 1  

print len(document_corpus.keys())

document_corpus_title = {}
document_corpus_body = {}
document_corpus_body_title = {}
idx = 0
for doc_id in document_corpus:
    text = document_corpus[doc_id]['title']
    begin = time.time()
    text1 = preprocess(text, stopwords, lemmatizer)
    end = time.time()
    #print ('title time: ',(end-begin))
    document_corpus_title[doc_id] = text1
    text = document_corpus[doc_id]['body']
    begin = time.time()
    text2 = preprocess(text, stopwords, lemmatizer)
    end = time.time()
    #print ('Body: ', (end-begin))
    document_corpus_body[doc_id] = text2
    text = text1 + ' ' + text2
    document_corpus_body_title[doc_id] = text
    idx += 1 
    if idx%100 ==0:
        print idx

Product_documents = document_corpus_title
with open("../Web_data/Product_documents_only_title.json","wb") as outfile:
	json.dump(document_corpus_title , outfile)
'''
tf_vectorizer = CountVectorizer(lowercase = True)
Product_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.fit_transform(Product_documents.values())
#pickle.dump( tf_vectorizer, open("../Web_data/tf_vectorizer_only_title.p","wb" ))
print ('Vocabulary: ', len(tf_vectorizer.get_feature_names()))    
'''
Product_documents = document_corpus_body
with open("../Web_data/Product_documents_only_body.json","wb") as outfile:
	json.dump(document_corpus_body , outfile)
'''
tf_vectorizer = CountVectorizer(lowercase = True)
Product_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.fit_transform(Product_documents.values())
#pickle.dump( tf_vectorizer, open("../Web_data/tf_vectorizer_only_body.p","wb" ))
print ('Vocabulary: ', len(tf_vectorizer.get_feature_names()))    
'''

Product_documents = document_corpus_body_title
with open("../Web_data/Product_documents_title_body.json","wb") as outfile:
	json.dump(document_corpus_body , outfile)
'''
tf_vectorizer = CountVectorizer(lowercase = True)
Product_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.fit_transform(Product_documents.values())
#pickle.dump( tf_vectorizer, open("../Web_data/tf_vectorizer_title_body.p","wb" ))
print ('Vocabulary: ', len(tf_vectorizer.get_feature_names()))    
'''