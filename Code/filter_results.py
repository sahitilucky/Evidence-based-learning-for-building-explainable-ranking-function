import io
import pickle

def get_string(value):
    strVal = value
    try:
        strVal = str(value)
    except:
        strVal = value.encode('utf-8')
    return strVal

doc_list = []
InputDirectory = '../Web_data/MQ2007/fold1/'
'''
product_documents = pickle.load(open("../../Web_data/Product_documents_corpus_sorted_2.p","rb"))
print len(product_documents.keys())
doc_list = product_documents.keys()
'''


'''
with io.open(InputDirectory+'train_division.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'train_division_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)



with io.open(InputDirectory+'test_division.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'test_division_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)


with io.open(InputDirectory+'new_test.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'new_test_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)

with io.open(InputDirectory+'test_valid.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'test_valid_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)

with io.open(InputDirectory+'vali.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'valid_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)


with io.open(InputDirectory+'test.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'test_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)

with io.open(InputDirectory+'train_train_division.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'train_train_division_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)

with io.open(InputDirectory+'train_valid_division.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'train_valid_division_filtered.txt', 'w',encoding= 'utf-8') as outfile:
        for line in infile:
            features = line.split('#')[0][:-2]
            #train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
            if document in doc_list:
                outfile.write(line)
'''

queries = {}
with open(InputDirectory+'07-million-query-topics.1-10000.txt', 'r') as infile:
	for line in infile:
		query_id = line.strip().split(':')[0]
		query = line.strip().split(':')[1]
		queries[query_id] = query
print len(queries.keys())

#file_names = ['test_filtered', 'valid_filtered', 'test_valid_filtered', 'test_division_filtered', 'new_test_filtered','train_division_filtered', 'train_valid_division_filtered', 'train_train_division_filtered']
#file_names = ['train_valid_division_filtered', 'train_train_division_filtered']
#file_names = ['train', 'test', 'vali', 'test_division', 'train_division', 'new_q_new_doc' ,'train_train_division', 'train_valid_division']
file_names = ['train_division_1', 'old_q_new_d', 'old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'combined_test_set', 'train_train_division_1', 'train_valid_division_1','train_division_1_fold1', 'train_division_1_fold12', 'train_division_1_fold123', 'train_division_1_fold1234', 'train_division_1_fold12345']
for filename in file_names:
    with io.open(InputDirectory+filename+'.txt', 'r',encoding= 'utf-8') as infile:
        gone = 0
        with io.open(InputDirectory+ filename + '_ideal_list2.txt', 'w', encoding= 'utf-8') as outfile:
            for line in infile:
                features = line.split('#')[0][:-2]
                query_id = features.split(' ')[1].split('qid:')[1]
                relevance = features.split(' ')[0]
                document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
                #print queries[query_id]
                #print document
                #print relevance
                try:
                    outfile.write(query_id) 
                    outfile.write(u',')
                    outfile.write(document)
                    outfile.write(u',')
                    outfile.write(relevance)
                    outfile.write(u'\n')
                    #outfile.write(get_string(queries[query_id]) + ',' + get_string(document) + ',' + get_string(relevance) + '\n')
                except:
                    gone += 1
                    continue
        print gone
'''
popularity = {}
with io.open(InputDirectory+'train.txt', 'r',encoding= 'utf-8') as infile:
    for line in infile:
        features = line.split('#')[0][:-2]
        pop = features.split(' ')[42].split('41:')[1]
        document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
        popularity[document] = float(pop)
with io.open(InputDirectory+'vali.txt', 'r',encoding= 'utf-8') as infile:
    for line in infile:
        features = line.split('#')[0][:-2]
        pop = features.split(' ')[42].split('41:')[1]
        document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
        popularity[document] = float(pop)
with io.open(InputDirectory+'test.txt', 'r',encoding= 'utf-8') as infile:
    for line in infile:
        features = line.split('#')[0][:-2]
        pop = features.split(' ')[42].split('41:')[1]
        document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
        popularity[document] = float(pop)

with open('../../Web_data/document_pagerank.csv', 'w') as outfile:
    for p in popularity:
        outfile.write(p+','+str(popularity[p])+'\n')
'''
