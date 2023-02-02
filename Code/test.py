import io
import random
'''
InputDirectory = '../Web_data/MQ2007/fold1/'
with io.open(InputDirectory+'train_division_1.txt', 'r',encoding= 'utf-8') as infile:
    with io.open(InputDirectory+'train_train_division_1.txt', 'w',encoding= 'utf-8') as outfile1:
        with io.open(InputDirectory+'train_valid_division_1.txt', 'w',encoding= 'utf-8') as outfile2:
            for line in infile:
                x = random.random()
                if (x>0.5):
                    outfile1.write(line)
                else:
                    outfile2.write(line)
'''

for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set','train_valid_division_1']:
	Inputfile1 = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_IRF_features_without_prior_wonr.txt'
	Inputfile2 = '../Web_data/MQ2007/fold1/' + s + '.txt'
	query_doc_features = {}
	with open(Inputfile1, 'r') as infile:
	    for line in infile:
	        line = line.strip()
	        features = line.split('#')[0][:-1]
	        #print features
	        query_id = features.split(' ')[1].split('qid:')[1]
	        relevance = features.split(' ')[0]
	        document = line.split('#')[1].split('product=')[1].split(' ')[0]
	        features_list = [f.split(':')[1] for f in features.split(' ')[2:]]
	        #print features_list
	        query_doc_features[(query_id,document)] = features_list
	        
	outputfile = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
	with open(Inputfile2, 'r') as infile:
	    with open(outputfile, 'w') as outfile:
	        for line in infile:
	            line = line.strip()
	            features = line.split('#')[0][:-2]
	            comment =  line.split('#')[1]
	            #print features
	            #print comment
	            query_id = features.split(' ')[1].split('qid:')[1]
	            relevance = features.split(' ')[0]
	            document = line.split('#')[1].split('docid = ')[1].split(' ')[0]
	            if (query_id,document) in query_doc_features:
	                outfile.write(features + ' ')
	                feature_string = ' '.join([str(47+idx)+':'+str(f) for idx,f in enumerate(query_doc_features[(query_id,document)])])
	                #print (feature_string)
	                outfile.write(feature_string + ' ')
	                outfile.write('#' + comment + '\n')
	            else:
	                print ('gone')
	                print (query_id,document)
        

'''
import json
infile = json.load(open('../Web_data/MQ2007/fold1/q_term_to_d_term_train_train_division.json', 'r'))
for doc_term in infile:
	sorted_ts = sorted(infile[doc_term].items(), key = lambda l :l[1], reverse=True)
	print (doc_term,sorted_ts)
'''
'''
InputDirectory = '../Web_data/MQ2007/fold1/'
with io.open(InputDirectory+'train_division_1.txt', 'r',encoding= 'utf-8') as infile:
	with io.open(InputDirectory+'train_division_1_fold1.txt', 'w',encoding= 'utf-8') as outfile1:
		with io.open(InputDirectory+'train_division_1_fold12.txt', 'w',encoding= 'utf-8') as outfile2:
			with io.open(InputDirectory+'train_division_1_fold123.txt', 'w',encoding= 'utf-8') as outfile3:
				with io.open(InputDirectory+'train_division_1_fold1234.txt', 'w',encoding= 'utf-8') as outfile4:
					with io.open(InputDirectory+'train_division_1_fold12345.txt', 'w',encoding= 'utf-8') as outfile5:
						for line in infile:
							x = random.sample([1,2,3,4,5],1)[0]
							if x<=1:
								outfile1.write(line)
							if x<=2:
								outfile2.write(line)	
							if x<=3:
								outfile3.write(line)
							if x<=4:
								outfile4.write(line)
							if x<=5:
								outfile5.write(line)
'''						




