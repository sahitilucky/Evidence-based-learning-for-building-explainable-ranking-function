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
from MoreUtils import *
'''
def preprocess(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if not w in stopwords]
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    text = ' '.join(lemmas)
    return text
'''
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
#change this function
def document_gen_probability(query_att_value_gen_vector, product_att_values, product_tokens, tf_att_values):
    product_att_values = np.asarray(product_att_values).reshape(-1)
    #print product_att_values.shape
    #print query_att_value_gen_vector.shape
    probability = np.dot(product_att_values, query_att_value_gen_vector)
    return probability


def document_gen_probability_new(rel_doc_model, product_att_counts_doc, tf_att_values):
    probability = 0
    indices = np.nonzero(np.array(product_att_counts_doc))[0]
    total_counts = np.sum(np.array(product_att_counts_doc))
    word_probabilities = []
    for idx in indices:
        try:
            probability += (float(product_att_counts_doc[idx])/float(total_counts))*math.log(rel_doc_model[tf_att_values[idx]])
            word_probabilities += [(idx,rel_doc_model[tf_att_values[idx]])]
            #print (tf_att_values[idx])
            #print (float(product_att_counts_doc[idx])/float(total_counts))
            #print (rel_doc_model[tf_att_values[idx]])
        except:
            probability += (float(product_att_counts_doc[idx])/float(total_counts))*math.log(0.000000001)
            word_probabilities += [(idx,0.000000001)]
        '''
        try:
            probability -= product_att_counts_doc[idx]*math.log(non_rel_doc_model[tf_att_values[idx]])
        except KeyError:
            print ('coming here')
            print (idx)
            print (tf_att_values[idx])
            probability -= product_att_counts_doc[idx]*math.log(0.000000001)
        '''
    return (probability,word_probabilities)



def document_gen_probability_only_non_rel(non_rel_doc_model, product_att_counts_doc, tf_att_values):
    probability = 0
    indices = np.nonzero(np.array(product_att_counts_doc))[0]
    total_counts = np.sum(np.array(product_att_counts_doc))
    for idx in indices:
        try:
            probability -= (float(product_att_counts_doc[idx])/float(total_counts))*math.log(non_rel_doc_model[tf_att_values[idx]])
        except KeyError:
            print ('coming here')
            print (idx)
            print (tf_att_values[idx])
            probability -= (float(product_att_counts_doc[idx])/float(total_counts))*math.log(0.000000001)   
    return probability

def query_gen_probability(document_query_word, query):
    words = list(set(query.split(' ')))
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    num_values = len(query_att_value_gen)
    probability = 0 
    word_probabilities = []
    for word in words:
        #print (word)
        p_w_q = float(word_frequencies[word])/float(query_length)
        try:
            probability += word_frequencies[word]*math.log(document_query_word[word])
            word_probabilities += [(word, document_query_word[word])]
            #print ('have the word')
        except:
            probability += word_frequencies[word]*math.log(0.000000001)
            word_probabilities += [(word,0.000000001)]
            #print ('does not have the word')    
    return (probability, word_probabilities)


def indirect_query_gen_proability(doc_word_to_query_word, query, tf_att_values, document_query_word, doc_term_evidences, query_term_to_doc_term):
    #words = list(set(query.split(' ')))
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    probability = 0 
    word_probabilities = []
    evidence_weight = []
    for word in word_frequencies:
        word_probability = 0
        doc_words = []
        try:
            doc_words = query_term_to_doc_term[word].keys()
        except:
            pass
        for doc_word in doc_words:
            if doc_word!=word:
                try:
                    proba = doc_word_to_query_word[doc_word][word]
                    word_probability +=  document_query_word[doc_word]*proba
                    evidence_weight += [doc_term_evidences[doc_word]]
                    #print ('coming here')
                except:
                    pass
        if word_probability != 0:
            probability += word_frequencies[word]*math.log(word_probability)
            word_probabilities += [(word,word_probability)]
        else:
            probability += word_frequencies[word]*math.log(0.000000001)
            word_probabilities += [(word,0.000000001)]
    avg_weight = 0
    try:
        avg_weight = float(sum(evidence_weight))/float(len(evidence_weight))
    except:
        pass
    return (probability, word_probabilities, avg_weight)

def indirect_document_gen_probability(query_term_to_doc_term, product_att_counts_doc, rel_doc_model, tf_att_values, query_term_evidences, doc_word_to_query_word):
    probability = 0
    indices = np.nonzero(np.array(product_att_counts_doc))[0]
    total_counts = np.sum(np.array(product_att_counts_doc))
    evidence_weight = []
    word_probabilities = []
    for idx in indices:
        word_probability = 0
        q_words = []
        try:    
            q_words = doc_word_to_query_word[tf_att_values[idx]].keys()
        except:
            pass
        for word in q_words:
            try:
                if word!=tf_att_values[idx]:
                    word_probability +=  rel_doc_model[word]*query_term_to_doc_term[word][tf_att_values[idx]]
                    evidence_weight += [query_term_evidences[word]]
            except:
                pass
        if word_probability != 0:
            probability += (float(product_att_counts_doc[idx])/float(total_counts))*math.log(word_probability)
            word_probabilities += [(idx,word_probability)]
        else:
            probability += (float(product_att_counts_doc[idx])/float(total_counts))*math.log(0.000000001)
            word_probabilities += [(idx,0.000000001)]
    avg_weight = 0
    try:
        avg_weight = float(sum(evidence_weight))/float(len(evidence_weight))
    except:
        pass
    return (probability, word_probabilities, avg_weight)


def relevance_score(product_att_values, tf_feature_names_index, query):
    words = list(set(query.split(' ')))
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    probability = 0
    word_probabilities = []
    for word in words:
        try:
            p_w_d = product_att_values[tf_feature_names_index[word]]
        except:
            p_w_d = 0.000001
        if p_w_d == 0:
            p_w_d = 0.000001
        p_w_q = float(word_frequencies[word])/float(query_length)
        probability += p_w_q*math.log(p_w_d)
        word_probabilities += [probability]
    return (probability,word_probabilities)

def get_feature_string(features):
    tags = ['qid'] + list(range(1, len(features) + 1, 1))
    #tags = list(range(1, len(features) + 1, 1))
    tag_features = [str(x) + ':' + str(y) for x, y in zip(tags, features)]
    
    final_string = ' '.join(tag_features).strip()
    
    return final_string



#THIRD COMPONENT with unigram combined
def third_component_unigram_combined(product_att_values, tf_att_values, rel_jud, document_query_word):

    #document_query_word = pickle.load(open(OutputDirectory+search_log_query_likelihood_model,'rb'))
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

    Product_topics_new = {}
    for productId in product_att_values:
        #print Product_documents[productId]
        #print productId
        if productId in document_query_word:
            #print len(document_query_word[productId].keys())
            Product_topics_new[productId] = get_distribution(document_query_word[productId], product_att_values[productId], tf_att_values, alpha3_parameter[productId])
        else:
            Product_topics_new[productId] = get_distribution({}, product_att_values[productId], tf_att_values, 0)
        #print len(Product_topics_new[productId].keys())

    print ('completed')
    return Product_topics_new
    #pickle.dump(Product_topics_new, open(OutputDirectory+search_log_unigram_query_likelihood_model,'wb'))


#inputfiles
ts = 'train_train_division_1'
popularityfile = '../Web_data/document_pagerank.csv'
product_word_documents_file = "../Web_data/Product_documents_title_body.json"
product_word_tf_vectorizer_file = "../Web_data/tf_vectorizer_title_body.p"

SearchlogDirectory = '../Web_data/MQ2007/fold1/'
search_log_query_document_probability = 'query_document_rel_probability_' + ts + '.p'
search_log_document_likelihood_model = 'query_att_value_gen_' + ts + '.json'
search_log_query_likelihood_model = 'document_query_word_' + ts + '.json'
query_term_translation_model = 'q_term_to_d_term_' + ts + '.json'
document_term_translation_model = 'd_term_to_q_term_' +ts+ '.json'
product_documents_probs = 'product_documents_word_probs'
product_documents_counts = 'product_documents_word_counts'

#search_log_unigram_query_likelihood_model = 'document_dist_unigram_search_log_train_testing.p'
#prior using popularity.
Popularity={}
with open(popularityfile,'r') as inputfile:
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

print ('loading...')
rel_jud = pickle.load(open(SearchlogDirectory+search_log_query_document_probability,'rb'))
document_query_word_search_log = json.load(open(SearchlogDirectory+search_log_query_likelihood_model,'r'))
#print (len(document_query_word_search_log.keys()))
#print (document_query_word_search_log.keys())
query_att_value_gen = json.load(open(SearchlogDirectory + search_log_document_likelihood_model, 'r'))
#print (len(query_att_value_gen.keys()))
#print (query_att_value_gen.keys())
query_term_translation_model = json.load(open(SearchlogDirectory + query_term_translation_model, 'r'))
for q_term in query_term_translation_model:
    query_term_translation_model[q_term] = {x:query_term_translation_model[q_term][x] for x in query_term_translation_model[q_term] if query_term_translation_model[q_term][x]>=0.001}
#print (len(query_term_translation_model.keys()))
#print (query_term_translation_model.keys())
print ('done1...')
document_term_translation_model = json.load(open(SearchlogDirectory + document_term_translation_model, 'r'))
for d_term in document_term_translation_model:
    document_term_translation_model[d_term] = {x:document_term_translation_model[d_term][x] for x in document_term_translation_model[d_term] if document_term_translation_model[d_term][x]>=0.01}
#print (len(document_term_translation_model.keys()))
#print (document_term_translation_model.keys())
print ('done2...')
#Product_att_value_tokens = json.load(open(product_word_documents_file,"rb"))
print ('done3...')
#product_tf_vectorizer = pickle.load(open(product_word_tf_vectorizer_file,"rb" ))
#tf_att_values = product_tf_vectorizer.get_feature_names()
#product_att_values = json.load(open(SearchlogDirectory + product_documents_probs, 'r'))
print ('done4...')
product_att_values_counts = json.load(open(SearchlogDirectory + product_documents_counts, 'r'))
print ('done5...')
#document_query_word_search_log_unigram = third_component_unigram_combined(product_att_values, tf_att_values, rel_jud, document_query_word)
non_rel_doc_model = dict(json.load(open("../Web_data/MQ2007/fold1/non_rel_doc_model.json","r")))
query_term_evidences = json.load(open(SearchlogDirectory + 'query_term_evidences_' + ts + '.json', 'r'))
#print (query_term_evidences)
doc_term_evidences = json.load(open(SearchlogDirectory + 'doc_term_evidences_' + ts + '.json', 'r')) 
#print (doc_term_evidences)
print ('done6...')
query_evidences = json.load(open(SearchlogDirectory + 'query_evidences_' + ts + '.json', 'r'))
doc_evidences = json.load(open(SearchlogDirectory + 'doc_evidences_' + ts + '.json', 'r'))
#print (query_evidences)
#print (doc_evidences)
print ('done7...')
tf_att_values = pickle.load(open('../Web_data/MQ2007/fold1/trimmed_vocabulary.p', 'rb'))
tf_att_values_index = dict(zip(tf_att_values, range(len(tf_att_values)))) 
print ('loading done...')

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
'''


def get_unigram_probabilities(query, productId):
    return relevance_score(Product_topics[productId], tf_att_values_index, query.lower(), Product_documents)

def get_unigram_model(product_att_counts_doc, tf_att_values):
    indices = np.nonzero(np.array(product_att_counts_doc))[0]
    total_counts = np.sum(np.array(product_att_counts_doc))
    unigram_language_model = {}
    for idx in indices:
        unigram_language_model[tf_att_values[idx]] = float(product_att_counts_doc[idx])/float(total_counts)  
    return unigram_language_model

def get_IRF_features(query, productId, qid, rel_jud2):
    #print ('taking time here-1')
    query2 = query
    query = preprocess(query)
    #print ('taking time here-0.5')
    features = []
    features.append(qid)
    #first component
    #return ([0],'0')
    gone = 0
    there = 0
    query_doc_weight = 0.7
    count = 0
    try:
        probability1 = rel_jud[qid][productId]['click_probability']
    except:
        probability1 = 0
    features.append(probability1)
    query_doc_probability = probability1
    #second component with searh log
    query_length = len(query.split(' '))
    word_frequencies = Counter(query.split(' '))
    query_doc_gen_without_L = {}
    #print ('taking time here-1')
    for word in word_frequencies:
        try:
            query_doc_gen_without_L[word] = float(word_frequencies[word])/float(query_length)
        except:
            pass
    doc_weight = 0
    query_weight = 0  
    combined_doc_probability = 0
    combined_query_probability = 0 
    
    try:
        probability1,w_ps1 = document_gen_probability_new(query_att_value_gen[qid], product_att_values_counts[productId], tf_att_values)
        #print ('coming here')
        non_rel_probability = document_gen_probability_only_non_rel(non_rel_doc_model, product_att_values_counts[productId], tf_att_values)
        #print ('taking time here0')
        #non_rel_probability = 0
        probability2,w_ps2 = document_gen_probability_new(query_doc_gen_without_L, product_att_values_counts[productId], tf_att_values)
        probability3,w_ps3,evidence_weight = indirect_document_gen_probability(query_term_translation_model, product_att_values_counts[productId], query_att_value_gen[qid], tf_att_values, query_term_evidences, document_term_translation_model)
        #probability2 = 0
        #evidence_weight = 0
        #combined_probability2 = combine(w_ps1,w_ps2, doc_gen_weights1, doc_gen_weights2)
        average_relevance = float(sum(query_evidences[qid]))/float(len(query_evidences[qid]))
        num_evidence = len(query_evidences[qid])
        max_evidence = max(query_term_evidences.values())
        weight1 = float(num_evidence)/float(5+num_evidence)
        weight3 = float(evidence_weight)/float(max_evidence)
        weight3 = float(weight3)/float(5+num_evidence+weight3)
        weight2 = float(1)/float(5+num_evidence)
        if (probability1) == 0:
            probability1 = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        if (non_rel_probability) == 0:
            non_rel_probability = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        if (probability3) == 0:
            probability3 = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        #print (probability1)
        #print ('Query', qid)
        #print ('productID', productId)
        #print ('Relevance', rel_jud2[qid][productId])
        combined_doc_probability = (weight1)*probability1 + weight3*probability3 #+ weight2*probability2#+ (weight3)*non_rel_probability
        if (weight1+weight3 != 0):
            combined_doc_probability = combined_doc_probability/(weight1+weight3)
        else:
            combined_doc_probability = combined_doc_probability            
        #combined_doc_probability = probability1 #+ (weight3)*non_rel_probability #+ (float(1)/float(10))*non_rel_probability
        doc_weight = 5 +num_evidence + float(evidence_weight)/float(max_evidence) #
        if (combined_doc_probability == 0):
            combined_doc_probability = math.log(0.000000001)
        if (evidence_weight !=0):
            count += 1 
        #    print ('Q_to_D Evidence weight',evidence_weight, weight2, probability2)
        features.append(probability1)
        #features.append(probability2)
        features.append(probability3)
        #features.append(average_relevance)
        features.append(num_evidence)
        features.append(evidence_weight)
        features.append(max_evidence)
        features.append(weight1)   
        features.append(weight3)          

        #print ('taking time here1')
    except:
        probability1,w_ps1 = document_gen_probability_new({}, product_att_values_counts[productId], tf_att_values)
        #probability1 = math.log(0.000000001)
        non_rel_probability = document_gen_probability_only_non_rel(non_rel_doc_model, product_att_values_counts[productId], tf_att_values)
        #print ('taking time here1.5')
        #non_rel_probability = 0
        probability2,w_ps2 = document_gen_probability_new(query_doc_gen_without_L, product_att_values_counts[productId], tf_att_values)
        probability3,w_ps3,evidence_weight = indirect_document_gen_probability(query_term_translation_model, product_att_values_counts[productId], query_doc_gen_without_L, tf_att_values, query_term_evidences, document_term_translation_model)        
        #probability2 = 0
        #evidence_weight = 0
        #combined_probability2 = probability2
        max_evidence = max(query_term_evidences.values())
        weight1 = 0
        weight3 = float(evidence_weight)/float(max_evidence)
        weight3 = float(weight3)/float(5+weight3)
        weight2 = float(1)/float(5)
        if (probability1) == 0:
            probability1 = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        if (non_rel_probability) == 0:
            non_rel_probability = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        if (probability3) == 0:
            probability3 = math.log(0.000000001)
            gone += 1
        else:
            there += 1
        
        combined_doc_probability = weight1*probability1 + weight3*probability3 #+ weight2*probability2#+ (0.5)*non_rel_probability
        if (weight1+weight3!= 0):
            combined_doc_probability = combined_doc_probability/(weight1+weight3) 
        else:
            combined_doc_probability = combined_doc_probability            
        #combined_doc_probability = probability1 #+ (0.9)*non_rel_probability
        doc_weight = 5 + float(evidence_weight)/float(max_evidence)
        if (combined_doc_probability == 0):
            combined_doc_probability = math.log(0.000000001)
        if (evidence_weight !=0):
            count += 1 
        
        #if (evidence_weight !=0):
        #    print ('Q_to_D Evidence weight',evidence_weight, weight2, probability2)
        features.append(probability1)
        #features.append(probability2)
        features.append(probability3)
        #features.append(0)
        features.append(0)
        features.append(evidence_weight)
        features.append(max_evidence)
        features.append(weight1)   
        features.append(weight3)          
    
        #print ('taking time here2')
    
    try:
        probability1,w_ps1 = query_gen_probability(document_query_word_search_log[productId], query)
        #print ('taking time here2.5')
        probability2,w_ps2 = query_gen_probability(get_unigram_model(product_att_values_counts[productId], tf_att_values), query)
        probability3,w_ps3,evidence_weight = indirect_query_gen_proability(document_term_translation_model, query, tf_att_values, document_query_word_search_log[productId], doc_term_evidences, query_term_translation_model) 
        #probability3 = 0
        #evidence_weight = 0
        average_relevance = float(sum(doc_evidences[productId]))/float(len(doc_evidences[productId]))
        num_evidence = len(doc_evidences[productId])
        max_evidence = max(doc_term_evidences.values())
        weight1 = float(num_evidence)/float(10+num_evidence)
        weight3 = float(evidence_weight)/float(max_evidence)
        weight3 = float(weight3)/float(5+num_evidence+weight3)
        weight2 = float(10)/float(10+num_evidence)
        combined_query_probability = weight1*probability1+weight2*probability2+weight3*probability3
        combined_query_probability = float(combined_query_probability)/float(weight1+weight2+weight3)
        #combined_query_probability = probability2
        #combined_query_probability = float(combined_query_probability)/float(weight2+weight3)
        query_weight = 5 +num_evidence + float(evidence_weight)/float(max_evidence) 
        if (evidence_weight !=0):
            count += 1 
        
        #if (evidence_weight !=0):
        #    print ('D_to_Q Evidence weight',evidence_weight, weight3, probability3)
        features.append(probability1)
        features.append(probability2)
        features.append(probability3)
        #features.append(average_relevance)
        features.append(num_evidence)   
        features.append(evidence_weight)
        features.append(max_evidence)  
        features.append(weight1)   
        features.append(weight2)
        features.append(weight3)          
        #print ('taking time here3')    
    except:
        probability1 = math.log(0.000000001)
        probability2,w_ps2 = query_gen_probability(get_unigram_model(product_att_values_counts[productId], tf_att_values), query)
        #print ('taking time here3.5')
        probability3,w_ps3,evidence_weight = indirect_query_gen_proability(document_term_translation_model, query, tf_att_values, get_unigram_model(product_att_values_counts[productId], tf_att_values), doc_term_evidences, query_term_translation_model)
        #probability3 = 0
        #evidence_weight = 0
        max_evidence = max(doc_term_evidences.values())
        weight1 = 0
        weight3 = float(evidence_weight)/float(max_evidence)
        weight3 = float(weight3)/float(5+weight3)
        weight2 = float(10)/float(10)
        combined_query_probability = weight1*probability1+weight2*probability2+weight3*probability3
        combined_query_probability = float(combined_query_probability)/float(weight1+weight2+weight3)
        #combined_query_probability = probability2
        #combined_query_probability = float(combined_query_probability)/float(weight2+weight3)
        query_weight =  5 + float(evidence_weight)/float(max_evidence)
        if (evidence_weight !=0):
            count += 1 
        
        #if (evidence_weight !=0):
        #    print ('D_to_Q Evidence weight',evidence_weight, weight3, probability3)
        features.append(math.log(0.000000001))
        features.append(probability2)
        features.append(probability3)
        #features.append(0)
        features.append(0) 
        features.append(evidence_weight)
        features.append(max_evidence) 
        features.append(weight1)   
        features.append(weight2)
        features.append(weight3)          

        #print ('taking time here4')
    '''
    if query in query_att_value_gen:
        probability2 = alpha2_parameter[query]*document_gen_probability(query_att_value_gen[query], product_att_values[productId], Product_att_value_tokens[productId], tf_att_values)
        #FeatureVector.append(str("{0:.6f}".format(document_gen_probability(query_att_value_gen[query], product_att_values[productId], Product_att_value_tokens[productId], tf_att_values))))
        features.append(str(alpha2_parameter[query])) 
        features.append(str(document_gen_probability(query_att_value_gen[query], product_att_values[productId], Product_att_value_tokens[productId], tf_att_values)))
    else:
        probability2 = 0
        features.append(str(0))
        features.append(str(0))
    #third component with search log and unigram
    try:
        probability3 = alpha4_parameter[query]*query_gen_probability(document_query_word_search_log_unigram[productId], query)
    except:
        probability3 = (float(1)/float(100))*query_gen_probability(document_query_word_search_log_unigram[productId], query)
    
    #third component only search log
    if productId in document_query_word_search_log:
        features.append(str(alpha3_parameter[productId]))
        features.append(str(query_gen_probability(document_query_word_search_log[productId], query)))
        
    else:
        features.append(str(0.0001))
        features.append(str(math.log(0.000000001)))

    #fourth component unigram
    try:
        features.append(str(alpha4_parameter[query])) 
        features.append(str(get_unigram_probabilities(query, productId)))
        
    except:
        features.append(str(float(1)/float(100))) 
        features.append(str(get_unigram_probabilities(query, productId)))
        
    probability = (alpha1_parameter*probability1) + (probability2) + (probability3) 
    
    try:
        probability = probability + alpha4_parameter[query]*add_prior(prior, productId) #- (0.5*negative_fb)
        features.append(str(add_prior(prior, productId)))
    except:
        probability = probability + (float(1)/float(100))*add_prior(prior, productId)
        features.append(str(add_prior(prior, productId)))
    features.append(str(probability))
    '''
    doc_weight = float(doc_weight)/float(10+doc_weight)
    query_weight = float(query_weight)/float(10+query_weight)
    #doc_weight*combined_doc_probability + 
    #query_weight*
    final_probability = doc_weight*combined_doc_probability + query_weight*combined_query_probability + query_doc_weight*query_doc_probability
    #final_probability = combined_query_probability
    features.append(final_probability)
    #print ('Doc weight, query weight ',doc_weight, query_weight, combined_doc_probability, combined_query_probability, rel_jud2[qid][productId])
    #print (final_probability)
    commentData = '#query='+get_string(qid)+',product='+get_string(productId)
    feature_string = get_feature_string(features)
    feature_string += ' '+ commentData + "\n"
    #print (gone, there)
    return (features,feature_string, count)

queries = {}
with open('../Web_data/MQ2007/fold1/07-million-query-topics.1-10000.txt', 'r') as infile:
    for line in infile:
        query_id = line.strip().split(':')[0]
        query = line.strip().split(':')[1]
        queries[query_id] = query
print len(queries.keys())



for s in  ['train_divsion_1']:#['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d']: #'combined_test_set']: #['vali','new_q_new_doc','test_division','train_valid_division','test']:
    #testing input files
    Input_test_file = s + '_ideal_list2.txt'
    IRF_features_file = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_IRF_features_without_prior_wonr.txt'
    perf_file = '../Web_data/search_log_results/search_log_ttd1_' + s + '_outer_comb_1_wonr.txt'
    #testing:
    rel_jud2 = {}
    with io.open(SearchlogDirectory + Input_test_file, 'r', encoding='utf-8') as Clickdata_list:
        for line in Clickdata_list:
            query,productId,relevance = line.strip('\n').split(',')
            relevance = float(relevance)
            try:
                rel_jud2[query][productId] = relevance
            except:
                rel_jud2[query] = {}
                rel_jud2[query][productId] = relevance

    print ('Num '+ s+ ' queiries', len(rel_jud2.keys()))
    LTRtrainingfile = open(IRF_features_file,'w') 
    ranked_docs = {}
    qid = 0
    evi_count = 0 
    total_count = 0
    for query in rel_jud2:
        #print ('QueryID: ', query)
        qid += 1
        probabilities = {}
        #print ('num docs: ', len(rel_jud2[query].keys()))
        for productId in rel_jud2[query]:
            #print ('productId, relevance', productId, rel_jud2[query][productId])
            LTRtrainingfile.write(str(rel_jud2[query][productId])+' ')
            (features,feature_string,count) = get_IRF_features(queries[query], productId, query, rel_jud2)
            LTRtrainingfile.write(feature_string)
            probabilities[productId] = float(features[-1])
            if count!=0:
                evi_count += 1
            total_count += 1
            #print (qid,productId)
        ranked_docs[query] = sorted(probabilities.items(), key = lambda x:x[1], reverse = True)
        #print ranked_docs[query][:10]
        #print (evi_count)
        if(qid%100 ==0):
            print (qid)
    print evi_count
    print total_count
    print ('Num '+ s+ ' queiries', len(rel_jud2.keys()))
    LTRtrainingfile.flush()
    LTRtrainingfile.close()


    filtered_rel_jud = rel_jud2
    filtered_ranked_docs = ranked_docs
    avg_measure_dict = {"ndcg1": 0.0, "ndcg3": 0.0, "ndcg5": 0.0, "ndcg10": 0.0, "p1": 0.0, "p3": 0.0, "p5": 0.0,
                            "p10": 0.0, "num_rel": 0.0}
    measures_list = ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "p1", "p3", "p5", "p10", "num_rel"]
    per_query_measures = {}
    for qid in filtered_ranked_docs.keys():
        #print ('QueryID:', qid)
        sorted_res = sorted(filtered_ranked_docs[qid], key = lambda x:x[1], reverse=True)
        per_query_measures[qid] = evaluate_res(sorted_res, filtered_rel_jud[qid])

        for m in per_query_measures[qid]:
            avg_measure_dict[m] += per_query_measures[qid][m]

    for m in avg_measure_dict:
        avg_measure_dict[m] /= float(len(filtered_ranked_docs))

    with open(perf_file, 'w') as output_file:
        output_file.write("qid,ndcg1,ndcg3,ndcg5,ndcg10,p1,p3,p5,p10,num\n")
        for qid in filtered_ranked_docs.keys():
            output_file.write(get_string(qid))
            for m in measures_list:
                output_file.write("," + str(per_query_measures[qid][m]))
            output_file.write("\n")
        output_file.write("all")
        for m in measures_list:
            output_file.write("," + str(avg_measure_dict[m]))






