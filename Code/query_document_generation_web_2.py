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
from third_component import *

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

def get_string(value):
    strVal = value
    try:
        strVal = str(value)
    except:
        strVal = value.encode('utf-8')
    return strVal

def add_prior(prior, productId):
    try:
        probability = math.log(prior[productId], 2)
    except:
        probability = math.log(0.0001,2)
    return probability
'''
def document_gen_probability(query_att_value_gen, product_att_values, product_tokens, tf_att_values):
    query_length = len(product_tokens.split(' '))
    indeces = np.nonzero(product_att_values)
    print indeces
    values = product_att_values[indeces]
    print values
    probability = 0 
    for index in indeces:
        p_w_q = (product_att_values[0,index])
        try:
            probability += p_w_q*math.log(query_att_value_gen[tf_att_values[index]])
        except:
            probability += p_w_q*math.log(0.000000001)
    return probability
'''
'''
def document_gen_probability(query_att_value_gen_vector, product_att_values, product_tokens, tf_att_values):
    query_length = len(product_tokens.split(' '))
    word_frequencies = Counter(product_tokens.split(' '))
    probability = 0 
    for word in word_frequencies:
        p_w_q = float(word_frequencies[word])/float(query_length)
        try:
            probability += p_w_q*math.log(query_att_value_gen[tf_att_values.index(word)])
        except:
            probability += p_w_q*math.log(0.000000001)
    return probability
'''
def document_gen_probability(query_att_value_gen_vector, product_att_values, product_tokens, tf_att_values):
    product_att_values = np.asarray(product_att_values).reshape(-1)
    #print product_att_values.shape
    #print query_att_value_gen_vector.shape
    probability = np.dot(product_att_values, query_att_value_gen_vector)
    return probability

def query_gen_probability(document_query_word, query):
    words = list(set(query.split(' ')))
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    num_values = len(query_att_value_gen)
    probability = 0 
    for word in words:
        #print (word)
        p_w_q = float(word_frequencies[word])/float(query_length)
        try:
            probability += p_w_q*math.log(document_query_word[word])
            #print ('have the word')
        except:
            probability += p_w_q*math.log(0.000000001)
            #print ('does not have the word')
    return probability

#InputDirectory1 = '../RawData/Furniture_weekWiseData/'
#InputDirectory2 = '../Data/Jan/Furniture/'

InputDirectory = '../../Web_data/MQ2007/fold1/'
OutputDirectory = '../../Web_data/Data/'

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


#document_gen = {}
queries = []
rel_jud = {}
queryID = 0
with io.open(InputDirectory + 'train_division_filtered_ideal_list2.txt', 'r', encoding='utf-8') as Clickdata_list:
    for line in Clickdata_list:
        query,productId,relevance = line.strip('\n').split(',')
        query = preprocess(query)
        if query not in queries:
            queries += [query]
            #test
            queryID += 1
        relevance = float(relevance)
        try:
            rel_jud[query][productId]={}
            rel_jud[query][productId]['relevance'] = relevance
        except:
            rel_jud[query] = {}
            rel_jud[query][productId]={}
            rel_jud[query][productId]['relevance'] = relevance
   

#new/Tokens_att_values_Jan_allweeks_furniture
print ('loading...')
Product_att_value_tokens = pickle.load(open("../../Web_data/Product_documents_corpus_sorted_2.p","rb"))

product_tf_vectorizer = pickle.load(open("../../Web_data/tf_vectorizer_corpus_sorted_2.p","rb" ))
product_att_value_id_order = Product_att_value_tokens.keys()
print len(Product_att_value_tokens.keys())
print 'doing transform'
product_att_values = product_tf_vectorizer.transform(Product_att_value_tokens.values())
tf_att_values = product_tf_vectorizer.get_feature_names()
print len(tf_att_values)
#P(E/S)
product_att_values = product_att_values.todense()
print np.sum(product_att_values, axis=1,dtype=float)
print np.isnan(np.sum(product_att_values))
print np.isnan(np.min(product_att_values))
print np.isnan(product_att_values).any()
#product_att_values = np.nan_to_num(product_att_values)
summation = np.sum(product_att_values, axis=1,dtype=float)
summation[summation==0] = 1
product_att_values = np.array(product_att_values, dtype=float)/summation
print np.isnan(np.sum(product_att_values))
print np.isnan(np.min(product_att_values))
print np.isnan(product_att_values).any()
print np.sum(product_att_values, axis=1,dtype=float)
product_att_values = dict(zip(product_att_value_id_order, product_att_values))
'''
query_att_value_gen = {}
for query in rel_jud:
    #print ('QueryID:', query)
    #query_att_value_gen[query] = {}
    for productId in rel_jud[query]:
        try:
            if rel_jud[query][productId]['relevance']>0:
                #print 'coming here'
                if query not in query_att_value_gen:
                    query_att_value_gen[query] = {}
                #print product_att_values[productId]
                #print product_att_values[productId].shape
        
                for (att_idx,att_value) in enumerate(tf_att_values):
                    try:
                        query_att_value_gen[query][att_idx] += product_att_values[productId][0, att_idx]*(float(rel_jud[query][productId]['relevance'])/float(2))
                        #if (product_att_values[productId][0, att_idx] !=0):
                            #print (product_att_values[productId][0, att_idx])
                            #print (float(rel_jud[query][productId]['relevance'])/float(2))

                    except:
                        query_att_value_gen[query][att_idx] = product_att_values[productId][0, att_idx]*(float(rel_jud[query][productId]['relevance'])/float(2))
                        #if query==u'linnmon / alex':
                        #    print product_att_values[productId][0, att_idx]
        except:
            continue
    if query in query_att_value_gen:
        total = sum(query_att_value_gen[query].values())
        print total
        print query
        for att_idx in query_att_value_gen[query]:
            if total == 0:
                query_att_value_gen[query][att_idx] = 0
            else:    
                query_att_value_gen[query][att_idx] = float(query_att_value_gen[query][att_idx])/float(total) 

print ('query att value gen done')
pickle.dump(query_att_value_gen, open(OutputDirectory+'query_att_value_train_train_division_filtered.p','wb'))
'''
#_Jan_1week_furniture
import json
dict_file = OutputDirectory + 'document_dist_unigram_search_log_train_division_filtered.json' 
with open(dict_file, 'r') as f:
    document_query_word = json.load(f)

#document_query_word = main_function()
print 'document query word loaded'
dict_file = OutputDirectory + 'query_att_value_train_division_filtered.json'
with open(dict_file, 'r') as f:
    query_att_value_gen = json.load(f)
print ('query att value loaded')
'''
query_att_value_gen = pickle.load(open(OutputDirectory+'query_att_value_train_division_filtered.p','rb'))
with open(OutputDirectory + 'query_att_value_train_division_filtered.json', 'w') as fp:
    json.dump(query_att_value_gen, fp)
'''
'''
document_query_word = {}
for query in rel_jud:
    query2 = query.replace('"','')
    #print query2
    words = list(set(query2.split(' ')))
    query_length = len(query2.split(' '))
    word_frequencies = Counter(query2.split(' '))
    for productId in rel_jud[query]:
        try:
            if rel_jud[query][productId]['relevance'] > 0:
                if productId not in document_query_word:
                    document_query_word[productId] = {}
                for word in words:
                    p_w_q = float(word_frequencies[word])/float(query_length)
                    try:
                        document_query_word[productId][word] += p_w_q*(float(rel_jud[query][productId]['relevance'])/float(2))
                    except:
                        document_query_word[productId][word] = p_w_q*(float(rel_jud[query][productId]['relevance'])/float(2))
        except:
            continue


for productId in document_query_word:
    total = sum(document_query_word[productId].values())
    for word in document_query_word[productId]:
        document_query_word[productId][word] = float(document_query_word[productId][word])/float(total)

print ('document done')
pickle.dump(document_query_word, open(OutputDirectory+'document_query_word_train_train_division_filtered.p','wb'))
'''


alpha4_parameter = {}
alpha1_parameter = 0.7
alpha2_parameter = {}
alpha3_parameter = {}
for query in rel_jud:
    alpha2_parameter[query] = []
    for productId in rel_jud[query]:
        if rel_jud[query][productId]['relevance'] > 0:
            alpha2_parameter[query] += [rel_jud[query][productId]['relevance']]
            try:
                alpha3_parameter[productId] += [rel_jud[query][productId]['relevance']]
            except:
                alpha3_parameter[productId] = [rel_jud[query][productId]['relevance']]
        #except:
        #    continue
    try:
        alpha2_parameter[query] = float(sum(alpha2_parameter[query]))/float(len(alpha2_parameter[query])) 
    except:
        alpha2_parameter[query] = 0
    alpha2_parameter[query] = float(alpha2_parameter[query])/float(100+alpha2_parameter[query])
    alpha4_parameter[query] = float(1)/float(100+alpha2_parameter[query])
for productId in alpha3_parameter:
    try:
        alpha3_parameter[productId] = float(sum(alpha3_parameter[productId]))/float(len(alpha3_parameter[productId])) 
    except:
        alpha3_parameter[productId] = 0
    alpha3_parameter[productId] = float(alpha3_parameter[productId])/float(10+alpha3_parameter[productId])
print ('parameters done')

#Component four
'''
LTR_predictions = pickle.load(open('../Models/LambdaMART/probabilistic_features/LTR_Jan_predictions.p','rb'))
print type(LTR_predictions)
print len(LTR_predictions)
print LTR_predictions[productId]
'''


#unigram_probabilities = pickle.load(open(OutputDirectory + 'Unigram_model_probabilities_test_filtered.p', 'rb'))
 

#testing:
rel_jud2 = {}
queries2 = []
queryID = 0
with io.open(InputDirectory + 'test_valid_filtered_ideal_list2.txt', 'r', encoding='utf-8') as Clickdata_list:
    for line in Clickdata_list:
        query,productId,relevance = line.strip('\n').split(',')
        
        if query not in queries2:
            queries2 += [query]
            #test
            queryID += 1
        relevance = float(relevance)
        try:
            rel_jud2[query][productId] = relevance
        except:
            rel_jud2[query] = {}
            rel_jud2[query][productId] = relevance


#ClickOutputfile = io.open(OutputDirectory+'junk.txt','w', encoding='utf-8')
#LTRtrainingfile = io.open(OutputDirectory+'Data/ClickData_prob_features_three_components_Jan_1week_all.txt','w', encoding='utf-8') 
LTRtrainingfile = io.open(OutputDirectory+'ClickData_prob_features_three_components_test_valid_filtered.txt','w', encoding='utf-8') 
ranked_docs = {}
qid = 0

for query in rel_jud2:
    print ('QueryID: ' + query)
    qid += 1
    probabilities = {}
    query2 = query
    query = preprocess(query)
    query_vector = []
    if query in query_att_value_gen:
        for (att_idx,word) in enumerate(tf_att_values):
            try:
                query_vector += [math.log(query_att_value_gen[query][att_idx])]
            except:
                query_vector += [math.log(0.000000001)]
    query_vector = np.array(query_vector)
    for productId in rel_jud2[query2]:
        print (productId)
        LTRtrainingfile.write(str(rel_jud2[query2][productId]).decode('utf-8')+' ')
        LTRtrainingfile.write('qid:'+str(qid).decode('utf-8')+' ')
        
        #print ('Product ID', productId)
        text = query +' ' + productId +' '
        try:
            probability1 = (float(rel_jud[query][productId]['relevance'])/float(2))
        except:
            probability1 = 0
        LTRtrainingfile.write('1:'+str(probability1).decode('utf-8') + ' ')
        text += 'probability1 ' + str(alpha1_parameter) + ' ' + str(probability1)+' '

        if query in query_att_value_gen:
            print ('are you taking time?1')
            probability2 = alpha2_parameter[query]*document_gen_probability(query_vector, product_att_values[productId], Product_att_value_tokens[productId], tf_att_values)
            text += 'probability2 ' + str(alpha2_parameter[query]) + ' ' + str(probability2)+' '
            #FeatureVector.append(str("{0:.6f}".format(document_gen_probability(query_att_value_gen[query], product_att_values[productId], Product_att_value_tokens[productId], tf_att_values))))
            LTRtrainingfile.write('2:'+str(alpha2_parameter[query]).decode('utf-8') + ' ' + '3:' + str(document_gen_probability(query_vector, product_att_values[productId], Product_att_value_tokens[productId], tf_att_values)).decode('utf-8')+' ')
            LTRtrainingfile.write('4:'+ str(probability2).decode('utf-8')+' ')
        else:
            text += 'probability2 ' + str(0) + ' ' + str(0)+' '
            #FeatureVector.append(str("{0:.6f}".format(0.0)))
            probability2 = 0
            LTRtrainingfile.write(u'2:0 3:0 4:0 ')
        
        try:
            print ('are you taking time?2')
            probability3 = alpha4_parameter[query]*query_gen_probability(document_query_word[productId], query)
            LTRtrainingfile.write('5:'+ str(alpha4_parameter[query]).decode('utf-8')+' 6:' + str(query_gen_probability(document_query_word[productId], query)).decode('utf-8')+' ')
            LTRtrainingfile.write('7:'+ str(probability3).decode('utf-8')+' ')
        except:
            print ('are you taking time?3')
            probability3 = (float(1)/float(100))*query_gen_probability(document_query_word[productId], query)
            LTRtrainingfile.write('5:'+str(float(1)/float(100)).decode('utf-8')+' 6:' + str(query_gen_probability(document_query_word[productId], query)).decode('utf-8')+' ')
            LTRtrainingfile.write('7:'+ str(probability3).decode('utf-8')+' ')
        
        probability = (alpha1_parameter*probability1) + (probability2) + (probability3) 
        #ClickOutputfile.write((text+'\n'))
        
        try:
            probability = probability #+ alpha4_parameter[query]*add_prior(prior, productId) #- (0.5*negative_fb)
            LTRtrainingfile.write('8:' + str(add_prior(prior, productId)).decode('utf-8')+' ')
            LTRtrainingfile.write('9:'+ str(alpha4_parameter[query]*add_prior(prior, productId)).decode('utf-8')+' ')
        except:
            probability = probability #+ (float(1)/float(100))*add_prior(prior, productId)
            LTRtrainingfile.write('8:' + str(add_prior(prior, productId)).decode('utf-8')+' ')
            LTRtrainingfile.write('9:'+ str((float(1)/float(100))*add_prior(prior, productId)).decode('utf-8')+' ')
        
        probabilities[productId] = probability
        LTRtrainingfile.write('#query='+query2+'\tproductId=' + productId + '\n')
    ranked_docs[query2] = sorted(probabilities.items(), key = lambda x:x[1], reverse = True)
    #print ranked_docs[query][:10]
print ('Num test queiries', len(queries2))
#ClickOutputfile.flush()
#ClickOutputfile.close()
LTRtrainingfile.flush()
LTRtrainingfile.close()

'''
ranked_docs = pickle.load(open('BM25-master/text/BM25_scores_Dec_train.p', 'r'))
filtered_ranked_docs = {}
for query in ranked_docs:
    filtered_ranked_docs[query] = []
    for product in ranked_docs[query]:
        if product[0] in rel_jud2[query]:
            filtered_ranked_docs[query] += [product]
'''

OutputDirectory = '../../Web_data/search_log_results/'
filtered_rel_jud = rel_jud2
filtered_ranked_docs = ranked_docs
avg_measure_dict = {"ndcg1": 0.0, "ndcg3": 0.0, "ndcg5": 0.0, "ndcg10": 0.0, "p1": 0.0, "p3": 0.0, "p5": 0.0,
                        "p10": 0.0, "num_rel": 0.0}
measures_list = ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "p1", "p3", "p5", "p10", "num_rel"]
per_query_measures = {}
for qid in filtered_rel_jud.keys():
    #print ('QueryID:', qid)
    sorted_res = sorted(filtered_ranked_docs[qid], key = lambda x:x[1], reverse=True)
    per_query_measures[qid] = evaluate_res(sorted_res, filtered_rel_jud[qid])

    for m in per_query_measures[qid]:
        avg_measure_dict[m] += per_query_measures[qid][m]

for m in avg_measure_dict:
    avg_measure_dict[m] /= float(len(filtered_rel_jud))

with open(OutputDirectory + 'search_log_test_valid_filtered_without_prior_2_3.txt', 'w') as output_file:
    output_file.write("qid,ndcg1,ndcg3,ndcg5,ndcg10,p1,p3,p5,p10,num\n")
    for qid in filtered_rel_jud.keys():
        output_file.write(get_string(qid))
        for m in measures_list:
            output_file.write("," + str(per_query_measures[qid][m]))
        output_file.write("\n")
    output_file.write("all")
    for m in measures_list:
        output_file.write("," + str(avg_measure_dict[m]))







