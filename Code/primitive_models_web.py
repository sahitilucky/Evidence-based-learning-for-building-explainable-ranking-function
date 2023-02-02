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
stopwords = stopwords.words()
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if not w in stopwords]
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    text = ' '.join(lemmas)
    return text

def relevance_score(Doc_word, tf_feature_names, query, Product_documents):
    query = preprocess(query)
    words = list(set(query.split(' ')))
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    #print word_frequencies
    probability = 0
    for word in words:
        #print word
        try:
            p_w_d = Doc_word[0,tf_feature_names.index(word.decode('utf-8'))]
            '''
            print ('found the word')
            print (Product_documents[productId])
            print (word)
            print Doc_word[0,tf_feature_names.index(word.decode('utf-8'))]
            '''
        except:
            p_w_d = 0.000001
        if p_w_d == 0:
        	p_w_d = 0.000001
        #print word_frequencies[word]
        #print query_length
        p_w_q = float(word_frequencies[word])/float(query_length)
        #print p_w_q
        probability += p_w_q*math.log(p_w_d)
    return probability

def add_prior(prior, productId):
    try:
        probability = math.log(prior[productId], 2)
    except:
        probability = math.log(0.0001,2)
    return probability


Product_documents = pickle.load(open("../../Web_data/Product_documents_corpus_sorted_2.p","rb"))
'''
tf_vectorizer = CountVectorizer(lowercase = True,min_df=5)
Product_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.fit_transform(Product_documents.values())
print len(tf_vectorizer.get_feature_names())
pickle.dump( tf_vectorizer, open("../../Web_data/tf_vectorizer_corpus_sorted_2.p","wb" ))
tf_feature_names = tf_vectorizer.get_feature_names()
pickle.dump( tf_feature_names, open("../../Web_data/tf_feature_names_corpus_sorted_2.p","wb" ) )
print len(tf_vectorizer.get_feature_names())
'''
tf_vectorizer = pickle.load(open("../../Web_data/tf_vectorizer_corpus_sorted_2.p","rb" ))
print len(Product_documents.keys())
tf_feature_names = tf_vectorizer.get_feature_names()
#print tf_feature_names
Porduct_id_order = Product_documents.keys()
Doc_word = tf_vectorizer.transform(Product_documents.values())
print 'transform done'
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

#prior using popularity.
Popularity={}
with open('../../Web_data/document_pagerank.csv','r') as inputfile:
    for line in inputfile:
        product=line.split(',')[0]
        pop= line.split(',')[1].strip('\n')
        pop = float(pop)
        Popularity[product]=pop
        
Popularity  = Popularity.items()
product_ids = map(lambda l :l[0], Popularity)
product_pops = map(lambda l :l[1], Popularity)
scaler = MinMaxScaler(feature_range=(0.0001,1))
popularity = scaler.fit_transform(np.reshape(np.array(product_pops), (-1,1))).reshape(1,-1).tolist()[0]
prior = dict(zip(product_ids, popularity))


InputDirectory = '../../Web_data/MQ2007/fold1/'
OutputDirectory = '../../Web_data/Data/'

queries = []
rel_jud = {}
queryID = 0
with io.open(InputDirectory + 'test_valid_filtered_ideal_list2.txt', 'r', encoding='utf-8') as Clickdata_list:
    for line in Clickdata_list:
        query,productId,relevance = line.strip('\n').split(',')
        #query = preprocess(query)
        if query not in queries:
            queries += [query]
            #test
            queryID += 1
        #if queryID<=1027:
        #    continue
        relevance = float(relevance)
        try:
            rel_jud[query][productId] = relevance
        except:
            rel_jud[query] = {}
            rel_jud[query][productId] = relevance

ranked_docs = {}

unigram_probabilities = {}
att_value_probabilities = {}
#att_values_att_value_level_probabilities = {}

for query in rel_jud:
    print ('QueryID:', query)
    #query = preprocess(query)
    probabilities = {}
    unigram_probabilities[preprocess(query)] = {}
    for productId in rel_jud[query]:
        probability1  = relevance_score(Product_topics[productId], tf_feature_names, query.lower(), Product_documents)
        probability = probability1 #+ add_prior(prior, productId)
        probabilities[productId] = probability
        unigram_probabilities[preprocess(query)][productId] = probability1
    ranked_docs[query] = sorted(probabilities.items(), key = lambda x:x[1], reverse = True)
pickle.dump(unigram_probabilities, open(OutputDirectory + 'Unigram_model_probabilities_test_valid_filtered.p', 'wb'))

filtered_rel_jud = rel_jud
filtered_ranked_docs = ranked_docs
avg_measure_dict = {"ndcg1": 0.0, "ndcg3": 0.0, "ndcg5": 0.0, "ndcg10": 0.0, "p1": 0.0, "p3": 0.0, "p5": 0.0,
                        "p10": 0.0, "num_rel": 0.0}
measures_list = ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "p1", "p3", "p5", "p10", "num_rel"]
per_query_measures = {}
for qid in filtered_rel_jud:
    sorted_res = sorted(filtered_ranked_docs[qid], key = lambda x:x[1], reverse=True)
    per_query_measures[qid] = evaluate_res(sorted_res, filtered_rel_jud[qid])

    for m in per_query_measures[qid]:
        avg_measure_dict[m] += per_query_measures[qid][m]

for m in avg_measure_dict:
    avg_measure_dict[m] /= float(len(filtered_rel_jud))

with open(OutputDirectory + 'Unigram_language_model_test_valid_without_prior_filtered.txt', 'w') as output_file:
    output_file.write("qid,ndcg1,ndcg3,ndcg5,ndcg10,p1,p3,p5,p10,num\n")
    for qid in filtered_rel_jud.keys():
        output_file.write(qid)
        for m in measures_list:
            output_file.write("," + str(per_query_measures[qid][m]))
        output_file.write("\n")
    output_file.write("all")
    for m in measures_list:
        output_file.write("," + str(avg_measure_dict[m]))




