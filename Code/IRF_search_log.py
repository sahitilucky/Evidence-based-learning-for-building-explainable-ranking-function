from MoreUtils import *
'''
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

#inputfiles

product_word_documents_file = "../Web_data/Product_documents_title_body.json"
product_word_tf_vectorizer_file = "../Web_data/tf_vectorizer_title_body.p"

#outputfiles
OutputDirectory = '../Web_data/MQ2007/fold1/'
for s in ['train_division_1', 'train_train_division_1', 'train_valid_division_1','train_division_1_fold1','train_division_1_fold123', 'train_division_1_fold1234','train_division_1_fold12']:
    print (s)
    Query_relevance_file = '../Web_data/MQ2007/fold1/' + s + '_ideal_list2.txt'
    search_log_query_document_probability = 'query_document_rel_probability_' + s + '.p'
    search_log_document_likelihood_model = 'query_att_value_gen_' + s + '.json'
    search_log_query_likelihood_model = 'document_query_word_' + s + '.json'
    #search_log_unigram_query_likelihood_model = 'document_dist_unigram_search_log_train_testing.p'
    query_term_translation_model = 'q_term_to_d_term_' + s + '.json'
    document_term_translation_model = 'd_term_to_q_term_' + s + '.json'
    product_documents_probs = 'product_documents_word_probs'
    product_documents_counts = 'product_documents_word_counts'
    #document_gen = {}
    rel_jud = {}
    with io.open(Query_relevance_file, 'r', encoding='utf-8') as Clickdata_list:
        for line in Clickdata_list:
            query,productId,relevance = line.strip('\n').split(',')
            relevance = float(relevance)
            try:
                rel_jud[query][productId]={}
                rel_jud[query][productId]['relevance'] = relevance
            except:
                rel_jud[query] = {}
                rel_jud[query][productId]={}
                rel_jud[query][productId]['relevance'] = relevance

    for query in rel_jud:
        for productId in rel_jud[query]:
            rel_jud[query][productId]['click_probability'] = float(rel_jud[query][productId]['relevance'])/float(2)
                
    print len(rel_jud.keys())
    print ('query product click probability done')
    pickle.dump(rel_jud, open(OutputDirectory+search_log_query_document_probability,'wb'))

    queries = {}
    with open('../Web_data/MQ2007/fold1/07-million-query-topics.1-10000.txt', 'r') as infile:
        for line in infile:
            query_id = line.strip().split(':')[0]
            query = line.strip().split(':')[1]
            queries[query_id] = query
    print len(queries.keys())

    '''
    #SECOND COMPONENT
    # product documents using Att-value pair  
    print ('loading...')
    Product_documents = json.load(open(product_word_documents_file,"rb"))

    #tf_vectorizer = CountVectorizer(lowercase = True, min_df = 5)
    tf_vectorizer= pickle.load(open(product_word_tf_vectorizer_file, 'rb'))
    product_att_value_id_order = Product_documents.keys()
    print ('Vocabulary: ', len(tf_vectorizer.get_feature_names()))   
    product_att_values = tf_vectorizer.fit_transform(Product_documents.values())
    tf_att_values = tf_vectorizer.get_feature_names()
    #pickle.dump(tf_vectorizer, open(product_word_tf_vectorizer_file, 'wb')) 

    #P(E/S)
    product_att_values_counts = product_att_values.todense()
    #print np.sum(product_att_values_counts, axis=1,dtype=float)
    #print np.isnan(np.sum(product_att_values_counts))
    #print np.isnan(np.min(product_att_values_counts))
    #print np.isnan(product_att_values_counts).any()
    #product_att_values = np.nan_to_num(product_att_values)
    summation = np.sum(product_att_values_counts, axis=1,dtype=float)
    summation[summation==0] = 1
    product_att_values = np.array(product_att_values_counts, dtype=float)/summation
    #print np.isnan(np.sum(product_att_values))
    #print np.isnan(np.min(product_att_values))
    #print np.isnan(product_att_values).any()
    #print np.sum(product_att_values, axis=1,dtype=float)
    product_att_values = np.array(product_att_values)
    print (product_att_values.shape)
    product_att_values = product_att_values.tolist()
    print (len(product_att_values))
    print (len(product_att_values[0]))
    print (product_att_values[0])
    product_att_values = dict(zip(product_att_value_id_order, product_att_values))
    print (product_att_values[product_att_value_id_order[0]])

    print (product_att_values_counts.shape)
    product_att_values_counts = product_att_values_counts.tolist()
    print (len(product_att_values_counts))
    print (len(product_att_values_counts[0]))
    print (product_att_values_counts[0])
    product_att_values_counts = dict(zip(product_att_value_id_order, product_att_values_counts))
    print (product_att_values_counts[product_att_value_id_order[0]])

    with open(OutputDirectory + product_documents_probs, 'w') as outfile:
        json.dump(product_att_values,  outfile)

    with open(OutputDirectory + product_documents_counts, 'w') as outfile:
        json.dump(product_att_values_counts, outfile)
    '''
    '''
    print ('loading')
    tf_att_values = pickle.load(open('../Web_data/MQ2007/fold1/trimmed_vocabulary.p', 'rb'))
    product_att_values = json.load(open(OutputDirectory + product_documents_probs, 'r'))
    product_att_values_counts = json.load(open(OutputDirectory + product_documents_counts, 'r'))
    print ('loading done...')
    
    query_att_value_gen = {}
    query_evidences = {}
    for query in rel_jud:
        #print ('QueryID:', query)
        #query_att_value_gen[query] = {}
        for productId in rel_jud[query]:
            try:
                if rel_jud[query][productId]['relevance']>0:
                    #print 'coming here'
                    try:
                        query_evidences[query] += [rel_jud[query][productId]['relevance']]
                    except:
                        query_evidences[query] = [rel_jud[query][productId]['relevance']]

                    if query not in query_att_value_gen:
                        query_att_value_gen[query] = {}
                    #print product_att_values[productId]
                    #print product_att_values[productId].shape
            
                    for (att_idx,att_value) in enumerate(tf_att_values):
                        try:
                            query_att_value_gen[query][att_value] += product_att_values[productId][att_idx]*(float(rel_jud[query][productId]['relevance'])/float(2))
                            #if (product_att_values[productId][0, att_idx] !=0):
                                #print (product_att_values[productId][0, att_idx])
                                #print (float(rel_jud[query][productId]['relevance'])/float(2))

                        except:
                            query_att_value_gen[query][att_value] = product_att_values[productId][att_idx]*(float(rel_jud[query][productId]['relevance'])/float(2))
                            #if query==u'linnmon / alex':
                            #    print product_att_values[productId][0, att_idx]
            except:
                continue
        if query in query_att_value_gen:
            total = sum(query_att_value_gen[query].values())
            #print total
            #print query
            for att_value in query_att_value_gen[query]:
                if total == 0:
                    query_att_value_gen[query][att_value] = 0
                else:    
                    query_att_value_gen[query][att_value] = float(query_att_value_gen[query][att_value])/float(total) 

    print ('query att value gen done')
    with open(OutputDirectory + search_log_document_likelihood_model, 'w') as outfile:
        json.dump(query_att_value_gen, outfile)



    doc_evidences = {}
    document_query_word = {}
    for query in rel_jud:
        query2 = preprocess(queries[query])
        #print query2
        words = list(set(query2.split(' ')))
        query_length = len(query2.split(' '))
        word_frequencies = Counter(query2.split(' '))
        for productId in rel_jud[query]:
            try:
                if rel_jud[query][productId]['relevance'] > 0:
                    try:
                        doc_evidences[productId] += [rel_jud[query][productId]['relevance']]
                    except:
                        doc_evidences[productId] = [rel_jud[query][productId]['relevance']]
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
    with open(OutputDirectory + search_log_query_likelihood_model, 'w') as outfile:
        json.dump(document_query_word, outfile)

    
    #Assume query vocabulary in small and subset of document vocabulary?
    #query_vocabulary = {}
    #term association count
    query_term_d_term = {}
    doc_term_q_term = {}
    query_term_evidences = {}
    doc_term_evidences = {word:0 for word in tf_att_values}
    for query in rel_jud:
        query2 = preprocess(queries[query])
        word_frequencies = Counter(query2.split(' '))
        for productId in rel_jud[query]:
            #print (query, productId, queries[query])
            indices = np.nonzero(np.array(product_att_values_counts[productId]))[0]
            #print (indices)
            #print (indices.shape)
            if rel_jud[query][productId]['relevance'] > 0:
                for index in indices:
                    doc_term_evidences[tf_att_values[index]] += 1
                for word in word_frequencies:
                    try:
                        query_term_evidences[word] += 1
                    except:
                        query_term_evidences[word] = 1
                    for index in indices:
                        #if (word != tf_att_values[index]):
                            #print ('coming here')
                        try:
                            query_term_d_term[word][tf_att_values[index]] += product_att_values_counts[productId][index]*word_frequencies[word] 
                        except:
                            try:
                                query_term_d_term[word][tf_att_values[index]] = product_att_values_counts[productId][index]*word_frequencies[word]
                            except:
                                query_term_d_term[word] = {}
                                query_term_d_term[word][tf_att_values[index]] = product_att_values_counts[productId][index]*word_frequencies[word]
                            #print ('here2')
                            #print word
                            #print tf_att_values[index]
                            #print (product_att_values_counts[productId][index]*word_frequencies[word])
                        try:
                            doc_term_q_term[tf_att_values[index]][word] += product_att_values_counts[productId][index]*word_frequencies[word]
                        except:
                            try:
                                doc_term_q_term[tf_att_values[index]][word] = product_att_values_counts[productId][index]*word_frequencies[word]
                            except:    
                                doc_term_q_term[tf_att_values[index]] = {}
                                doc_term_q_term[tf_att_values[index]][word] = product_att_values_counts[productId][index]*word_frequencies[word]
                            #print ('here4')
                                #print word
                                #print tf_att_values[index]
                                #print (product_att_values_counts[productId][index]*word_frequencies[word])
    length = 0
    for word in query_term_d_term:
        total_counts = sum(query_term_d_term[word].values()) 
        length += len(query_term_d_term[word].keys())
        #print (word)
        #print (query_term_d_term[word].keys())
        #print (query_term_d_term[word].values())
        for doc_word in query_term_d_term[word]:
            query_term_d_term[word][doc_word] = float(query_term_d_term[word][doc_word])/float(total_counts)

    length2 = 0
    for doc_word in doc_term_q_term:
        total_counts = sum(doc_term_q_term[doc_word].values()) 
        length2 += len(doc_term_q_term[doc_word].keys())
        #print (doc_word)
        #print (doc_term_q_term[doc_word].keys())
        #print (doc_term_q_term[doc_word].values())
        for word in doc_term_q_term[doc_word]:
            doc_term_q_term[doc_word][word] = float(doc_term_q_term[doc_word][word])/float(total_counts)

    print ('Number of q terms', len(query_term_d_term.keys()))
    print ('Total length:' , length)
    print ('Number of d terms', len(doc_term_q_term.keys()))
    print ('Total length:' , length2)

    with open(OutputDirectory + query_term_translation_model, 'w') as outfile:
        json.dump(query_term_d_term, outfile)

    with open(OutputDirectory + document_term_translation_model, 'w') as outfile:
        json.dump(doc_term_q_term, outfile)



    with open(OutputDirectory + 'query_term_evidences_' + s +'.json', 'w') as outfile:
        json.dump(query_term_evidences, outfile)

    with open(OutputDirectory + 'doc_term_evidences_' + s + '.json', 'w') as outfile:
        json.dump(doc_term_evidences, outfile)


    with open(OutputDirectory + 'query_evidences_' + s + '.json', 'w') as outfile:
        json.dump(query_evidences, outfile)

    with open(OutputDirectory + 'doc_evidences_' + s + '.json', 'w') as outfile:
        json.dump(doc_evidences, outfile)

    '''


            



            





