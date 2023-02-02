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
from LTRUtils import evaluate_res
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def get_distribution2(Doc_word, tf_feature_names):
    dist = {}
    for idx,name in enumerate(tf_feature_names):
        dist[name] = Doc_word[0,idx]
    return dist

def get_distribution(document_query_word, Doc_word, tf_feature_names, alpha3_parameter):
    dist = {}
    for idx,word in enumerate(tf_feature_names):
    	if Doc_word[0,idx]!=0:
	        dist[word] = (1-alpha3_parameter)*Doc_word[0,idx]
    for word in document_query_word:
        #print word
        if word in dist:
        #   print dist[word]
            dist[word] += alpha3_parameter*document_query_word[word]
        else:
        #    print 0
            dist[word] = alpha3_parameter*document_query_word[word]
        #print dist[word]
    return dist
def main_function():
    #unigram language model    
    Product_documents = pickle.load(open("../../Web_data/Product_documents_corpus_sorted_2.p","rb"))
    tf_vectorizer = pickle.load(open("../../Web_data/tf_vectorizer_corpus_sorted_2.p","rb" ))
    tf_feature_names = tf_vectorizer.get_feature_names()
    print tf_feature_names
    Porduct_id_order = Product_documents.keys()
    Doc_word = tf_vectorizer.transform(Product_documents.values())
    Doc_word = Doc_word.todense()
    #Background_dist = np.sum(Doc_word, axis = 0, dtype=float)
    #Background_dist = np.array(Background_dist, dtype=float)/float(np.sum(Background_dist))
    summation = np.sum(Doc_word, axis=1,dtype=float)
    summation[summation==0] = 1
    Doc_word = np.array(Doc_word,dtype=float)/summation
    print np.isnan(np.sum(Doc_word))
    print np.isnan(np.min(Doc_word))
    print np.isnan(Doc_word).any()
    gamma = 1.0
    Doc_word = (gamma)*Doc_word #+ (1-gamma)*np.repeat(Background_dist, Doc_word.shape[0], axis=0)
    Product_topics = dict(zip(Porduct_id_order, Doc_word))
    print ('done with product topics')


    OutputDirectory = '../../Web_data/Data/'
    document_query_word = pickle.load(open(OutputDirectory+'document_query_word_train_division_filtered.p','rb'))


    InputDirectory = '../../Web_data/MQ2007/fold1/'
    OutputDirectory = '../../Web_data/Data/'
    queries = []
    rel_jud = {}
    queryID = 0
    with io.open(InputDirectory + 'train_division_filtered_ideal_list2.txt', 'r', encoding='utf-8') as Clickdata_list:
        for line in Clickdata_list:
            query,productId,relevance = line.strip('\n').split(',')
            #query = preprocess(query)
            if query not in queries:
                queries += [query]
                #test
                queryID += 1
            #if queryID>1027:
            #    continue
            relevance = float(relevance)
            try:
                rel_jud[query][productId]={}
                rel_jud[query][productId]['relevance'] = relevance
            except:
                rel_jud[query] = {}
                rel_jud[query][productId]={}
                rel_jud[query][productId]['relevance'] = relevance


    alpha3_parameter = {}
    for query in rel_jud:
        for productId in rel_jud[query]:
            if rel_jud[query][productId]['relevance'] > 0:
                try:
                    alpha3_parameter[productId] += [rel_jud[query][productId]['relevance']]
                except:
                    alpha3_parameter[productId] = [rel_jud[query][productId]['relevance']]


    for productId in alpha3_parameter:
        if 0 in alpha3_parameter[productId]:
            print ("yes")
        alpha3_parameter[productId] = float(sum(alpha3_parameter[productId]))/float(len(alpha3_parameter[productId])) 
        alpha3_parameter[productId] = float(alpha3_parameter[productId])/float(10+alpha3_parameter[productId])

    tf_feature_names_new = tf_feature_names[:]
    for productId in document_query_word:
        for word in document_query_word[productId]:
            if word not in tf_feature_names:
                tf_feature_names_new += [word]

    Product_topics_new = {}
    for productId in Product_topics:
        #print Product_documents[productId]

        print productId
        if productId in document_query_word:
            print len(document_query_word[productId].keys())
            Product_topics_new[productId] = get_distribution(document_query_word[productId], Product_topics[productId], tf_feature_names, alpha3_parameter[productId])
        else:
            Product_topics_new[productId] = get_distribution({}, Product_topics[productId], tf_feature_names, 0)
        print len(Product_topics_new[productId].keys())
        '''
        dist = []
        for word in tf_feature_names_new:
            try:
                dist += [Product_topics_new[productId][word]]
                #print (word, Product_topics_new[productId][word])
            except:
                dist += [0]
        Product_topics_new[productId] = dist
     	
        for word in Product_topics_new[productId]:
            print (word,Product_topics_new[productId][word])
        '''
    print ('completed')
    return Product_topics_new
#pickle.dump(Product_topics_new, open(OutputDirectory+'document_dist_unigram_search_log_train_division_filtered.p','wb'))
#pickle.dump(tf_feature_names_new, open(OutputDirectory+'tf_feature_names_with_search_log_train_division_filtered    .p','wb'))
'''
product_topics_new = main_function()
OutputDirectory = '../../Web_data/Data/'
import json
with open(OutputDirectory + 'document_dist_unigram_search_log_train_division_filtered.json', 'w') as fp:
    json.dump(product_topics_new, fp)
'''