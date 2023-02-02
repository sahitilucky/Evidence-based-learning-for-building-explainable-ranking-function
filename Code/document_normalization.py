from MoreUtils import *

#GENERATING NON relevant document
product_word_documents_file = "../Web_data/Product_documents_title_body.json"
product_word_tf_vectorizer_file = "../Web_data/tf_vectorizer_title_body.p"
product_documents_probs = 'product_documents_word_probs_new'
product_documents_counts = 'product_documents_word_counts_new'
OutputDirectory = '../Web_data/MQ2007/fold1/'

def term_binary_frequency(doc_list, term):
	return len(filter(lambda l : term in l, doc_list))

def updated_frequency(df_t, N, V_t, V, factor):
	return float(df_t- V_t + factor)/float(N-V+(2*factor))

tf_vectorizer = pickle.load(open(product_word_tf_vectorizer_file, 'rb'))
documents = json.load(open(product_word_documents_file,"rb"))
documents_id_order = documents.keys()
document_matrix = tf_vectorizer.fit_transform(documents.values())
Doc_word = document_matrix.todense()
vocabulary = tf_vectorizer.get_feature_names()
with io.open('../Web_data/MQ2007/fold1/vocabulary.txt', 'w', encoding='utf-8') as outfile:
	for w in vocabulary:
		outfile.write(w + '\n')

term_u_t = [0]*len(vocabulary)
N = Doc_word.shape[0]
total_sum = 0
term_frequency = {}
for (idx,word) in enumerate(vocabulary):
	#print (np.transpose(np.nonzero(Doc_word[:,idx])).shape)
	#print (Doc_word[np.nonzero(Doc_word[:,idx])[0],idx])
	df_t = np.transpose(np.nonzero(Doc_word[:,idx])).shape[0]
	term_frequency[idx] = np.sum(Doc_word[:,idx])
	u_t = float(df_t)/float(N)	
	term_u_t[idx] = u_t
	total_sum += u_t
	term_frequency[idx] = term_frequency[idx]*math.log(float(N)/float(df_t+1))
term_u_t = [float(l)/float(total_sum) for l in term_u_t]
term_u_t = zip(vocabulary,term_u_t)
#json.dump(term_u_t, open("../Web_data/MQ2007/fold1/non_rel_doc_model.json","w"))



#prune the list and documents list accordingly
term_frequency = sorted(term_frequency.items(), key = lambda l: l[1], reverse=True)
term_frequency_values = [x[1] for x in term_frequency]
hist = np.histogram(term_frequency_values,bins=[1,10,100,1000,10000])
print(hist)
print (len(vocabulary))
print ("percentage of 8000 word: ", float(sum(term_frequency_values[:8000]))/float(sum(term_frequency_values)))
print ("percentage of 10000 word: ", float(sum(term_frequency_values[:10000]))/float(sum(term_frequency_values)))
print ("percentage of 100000 word: ", float(sum(term_frequency_values[:100000]))/float(sum(term_frequency_values)))
selected_indices = [x[0] for x in term_frequency[:10000]]
with io.open('../Web_data/MQ2007/fold1/trimmed_vocabulary_new.txt', 'w', encoding ='utf-8') as outfile:
	for w in term_frequency[:10000]:
		outfile.write(vocabulary[w[0]] + ' ' + unicode(w[1]) + '\n')
trimmed_vocabulary = [vocabulary[x] for x in selected_indices]
pickle.dump(trimmed_vocabulary, open('../Web_data/MQ2007/fold1/trimmed_vocabulary_new.p', 'wb'))

product_att_value_id_order = documents_id_order
product_att_values = Doc_word
#pickle.dump(tf_vectorizer, open(product_word_tf_vectorizer_file, 'wb')) 

#P(E/S)
print (vocabulary[selected_indices[0]])
sample = product_att_values[:,selected_indices[0]]
product_att_values_counts = product_att_values[:,selected_indices]
print (trimmed_vocabulary[0])
sample2 = product_att_values_counts[:,0]
print (sample==sample2).all()
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
#print (product_att_values[0])
product_att_values = dict(zip(product_att_value_id_order, product_att_values))
#print (product_att_values[product_att_value_id_order[0]])

print (product_att_values_counts.shape)
product_att_values_counts = product_att_values_counts.tolist()
print (len(product_att_values_counts))
print (len(product_att_values_counts[0]))
#print (product_att_values_counts[0])
product_att_values_counts = dict(zip(product_att_value_id_order, product_att_values_counts))
#print (product_att_values_counts[product_att_value_id_order[0]])

with open(OutputDirectory + product_documents_probs, 'w') as outfile:
    json.dump(product_att_values,  outfile)

with open(OutputDirectory + product_documents_counts, 'w') as outfile:
    json.dump(product_att_values_counts, outfile)


for i in range(100):
	print (trimmed_vocabulary[i], term_u_t[i])











