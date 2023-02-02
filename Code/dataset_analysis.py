InputDirectory = '../Web_data/MQ2007/fold1/'
documents_list = []
idx = 0
import io
import numpy as np
all_train_documents = []
train_query_ids = []
with io.open(InputDirectory+'train.txt', 'r',encoding= 'utf-8') as infile:
	for line in infile:
		features = line.split('#')[0][:-2]
		train_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
		document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
		#print (document)
		idx+= 1
		all_train_documents += [document]
		if document not in documents_list:
			documents_list += [document]
print idx
train_query_ids = list(set(train_query_ids))


all_test_documents = []
idx = 0
test_query_ids = []
with io.open(InputDirectory+'test.txt', 'r',encoding= 'utf-8') as infile:
	for line in infile:
		features = line.split('#')[0][:-2]
		test_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
		document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
		#print (document)
		idx += 1
		all_test_documents += [document]
		if document not in documents_list:
			documents_list += [document]
print idx
test_query_ids = list(set(test_query_ids))


idx = 0
valid_query_ids = []
all_valid_documents = []
with io.open(InputDirectory+'vali.txt', 'r',encoding= 'utf-8') as infile:
	for line in infile:
		features = line.split('#')[0][:-2]
		valid_query_ids += [int(features.split(' ')[1].split('qid:')[1])] 
		document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
		#print (document)
		idx += 1 
		all_valid_documents += [document]
		if document not in documents_list:
			documents_list += [document]
print idx
valid_query_ids = list(set(valid_query_ids))

print ('Train queries: ', len(train_query_ids))
print ('Test queries: ', len(test_query_ids))
print ('Valid queries: ', len(valid_query_ids))

inter = 0
for query_id in test_query_ids:
	if query_id in train_query_ids:
		print query_id
		print 'omg train test'
		inter += 1
print ('Train test queries: ',inter)
inter = 0 
for query_id in valid_query_ids:
	if query_id in train_query_ids:
		print query_id
		print 'omg valid train'
		inter += 1
print ('Train valid queries: ',inter)
inter = 0
for query_id in test_query_ids:
	if query_id in valid_query_ids:
		print query_id
		print 'omg test_valid'
		inter += 1
print ('Valid test queries: ',inter)
'''
with io.open(InputDirectory+'documents_list.txt', 'w') as outfile:
	for d in documents_list:
		outfile.write(d+'\n')
print documents_list[:20]
'''
all_valid_documents = list(set(all_valid_documents))
all_test_documents = list(set(all_test_documents))
all_train_documents = list(set(all_train_documents))
print ('Train documents:', len(all_train_documents))
print ('Test documents:', len(all_test_documents))
print ('Valid documents:', len(all_valid_documents))
print 'Valid and test document intersection:'
print len(set(all_valid_documents).intersection(set(all_test_documents)))
print 'Train and test document intersection:'
print len(set(all_train_documents).intersection(set(all_test_documents)))
print 'Train and Valid document intersection:'
print len(set(all_train_documents).intersection(set(all_valid_documents)))

#print len(set(all_test_documents).intersection(set(all_train_documents)))

all_train_documents = list(set(all_train_documents))
length = len(all_train_documents)
all_train_documents = np.random.permutation(all_train_documents).tolist()
train_division = all_train_documents[:int(2*(length/3))]
test_division = all_train_documents[int(2*(length/3)):]
print 'making division'
print len(train_division)
print len(test_division)

#print len(set(all_test_documents).intersection(set(train_division)))
print len(set(train_division).intersection(set(test_division)))

test_division_query_ids = []
train_division_query_ids = []
test_division_lines = {}
with io.open(InputDirectory+'train.txt', 'r',encoding= 'utf-8') as infile:
	with io.open(InputDirectory+'train_division_1.txt', 'w',encoding= 'utf-8') as outfile1:
		with io.open(InputDirectory+'test_division_1.txt', 'w',encoding= 'utf-8') as outfile2:
			for line in infile:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
				if document in train_division:
					outfile1.write(line)
					train_division_query_ids += [query_id]
				if document in test_division:
					test_division_query_ids += [query_id]
					if query_id in test_division_lines:
						test_division_lines[query_id] += [line]
					else:
						test_division_lines[query_id] = [line]
					outfile2.write(line)

test_division_query_ids = list(set(test_division_query_ids))
train_division_query_ids = list(set(train_division_query_ids))

'''
with io.open(InputDirectory+'test.txt', 'r',encoding= 'utf-8') as infile1:
	with io.open(InputDirectory+'vali.txt', 'r',encoding= 'utf-8') as infile2:
		with io.open(InputDirectory+'test_valid.txt', 'w',encoding= 'utf-8') as outfile:
			for line in infile1:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				if query_id in test_division_lines:
					test_division_lines[query_id] += [line]
				else:
					test_division_lines[query_id] = [line]
				outfile.write(line)
			for line in infile2:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				if query_id in test_division_lines:
					test_division_lines[query_id] += [line]
				else:
					test_division_lines[query_id] = [line]
				outfile.write(line)
'''
new_test_queries = list(set(test_division_query_ids + test_query_ids+valid_query_ids))

print len(train_division_query_ids)
print len(test_division_query_ids)

print ('Analysis ')
print ('Test division:')
print (len(test_division_query_ids))
print len(set(test_division_query_ids).intersection(set(train_division_query_ids)))
print len(set(test_division).intersection(set(train_division)))
print ('Analysis ')
print ('Test queries:')
print (len(test_query_ids))
print len(set(test_query_ids).intersection(set(train_division_query_ids)))
print len(set(all_test_documents).intersection(set(train_division)))

print ('Analysis ')
print ('Valid queries:')
print (len(valid_query_ids))
print len(set(valid_query_ids).intersection(set(train_division_query_ids)))
print len(set(all_valid_documents).intersection(set(train_division)))

new_query_lines = {}
new_queries = []
for query_id in new_test_queries:
	if query_id not in train_division_query_ids:
		new_queries += [query_id]
		new_query_lines[query_id] = []

print ('Analysis ')
print ('New queries:')
print (len(new_queries))
print len(set(new_queries).intersection(set(train_division_query_ids)))
		
pairs = []
repeats = 0
with io.open(InputDirectory+'test.txt', 'r',encoding= 'utf-8') as infile1:
	with io.open(InputDirectory+'vali.txt', 'r',encoding= 'utf-8') as infile2:
		with io.open(InputDirectory+'test_division_1.txt', 'r',encoding= 'utf-8') as infile3:
			for line in infile1:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
				if query_id in new_queries:
					if document in test_division:
						pairs += [(query_id, document)]
						new_query_lines[query_id] += [line]
			for line in infile2:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
				if query_id in new_queries:
					if document in test_division:
						if (query_id, document) not in pairs:
							new_query_lines[query_id] += [line]
						else:
							repeats += 1 
			for line in infile3:
				features = line.split('#')[0][:-2]
				query_id = int(features.split(' ')[1].split('qid:')[1])
				document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
				if query_id in new_queries:
					if document in test_division:
						if (query_id, document) not in pairs:
							new_query_lines[query_id] += [line]
						else:
							repeats += 1	

print ('Number of repetations: ', repeats)
with io.open(InputDirectory+'new_q_new_doc_1.txt', 'w',encoding= 'utf-8') as outfile2:
	for query in new_query_lines:
		for line in new_query_lines[query]:
			outfile2.write(line)
	

'''
with io.open(InputDirectory+'new_test.txt', 'w',encoding= 'utf-8') as outfile:
	for query_id in test_division_lines:
		for line in test_division_lines[query_id]:
			outfile.write(line)

with io.open(InputDirectory+'test_division_docids.txt', 'w',encoding= 'utf-8') as outfile:
	for line in test_division:
		outfile.write(line + '\n')


with io.open(InputDirectory+'train_division_docids.txt', 'w',encoding= 'utf-8') as outfile:
	for line in train_division:
		outfile.write(line + '\n')
'''
'''
length = len(train_division)
train_train_division = train_division[:int(2*(length/3))]
train_valid_division = train_division[int(2*(length/3)):]
with io.open(InputDirectory+'train_division.txt', 'r',encoding= 'utf-8') as infile:
	with io.open(InputDirectory+'train_train_division.txt', 'w',encoding= 'utf-8') as outfile1:
		with io.open(InputDirectory+'train_valid_division.txt', 'w',encoding= 'utf-8') as outfile2:
			for line in infile:
				document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
				if document in train_train_division:
					outfile1.write(line)
				if document in train_valid_division:
					outfile2.write(line)
'''
