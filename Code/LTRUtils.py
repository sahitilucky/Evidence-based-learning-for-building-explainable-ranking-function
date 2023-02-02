import Utils
import os
from PreProcessor import *
import operator
import numpy as np
import math
import io
import pickle
def learn_model(data_dir, model_dir, features_list, norm=None, ranker=6, metric="ERR@100"):
    # ranker:  0:MART, 1:RankNet, 2:RankBoost, 3:AdaRank, 4:CoordinateAscent, 6:LambdaMART, 7:ListNet, 8:RandomForests
    # norm: none, sum, zscore, linear
    # metric: MAP, NDCG @ k, DCG @ k, P @ k, RR @ k, ERR @ k

    norm_cmd = ""
    if norm is not None:
        norm_cmd = " -norm " + norm

    Utils.write_lines_to_file("features.temp", features_list)
    cmd = "java -jar ../RankLib/RankLib.jar -train " + data_dir + " -ranker " + str(ranker)
    cmd += " -feature features.temp -metric2t " + metric + norm_cmd + " -save " + model_dir
    os.system(cmd)
    print "Learning model.."
    os.system("rm features.temp")
    print "Finished learning."


def predict(model_dir, test_dir, norm=None):

    predictions = []
    norm_cmd = ""
    if norm is not None:
        norm_cmd = "-norm " + norm

    cmd = "java -jar ../RankLib/RankLib.jar -load " + model_dir
    cmd += " -rank " + test_dir + " -score  temp.prediction " + norm_cmd

    os.system(cmd)
    lines = Utils.read_lines_from_file("temp.prediction")
    for line in lines:
        predictions.append(float(line.split("\t")[2]))

    os.system("rm temp.prediction")
    return predictions


def evaluate_res(sorted_res, rel_jud):
    p5 = 0
    p10 = 0
    p1 = 0
    p3 = 0
    num_rel = 0

    for rank, tup in enumerate(sorted_res):
        try:
            if rel_jud[tup[0]] > 0:
                if rank < 1:
                    p1 += 1
                if rank < 3:
                    p3 += 1
                if rank < 5:
                    p5 += 1
                if rank < 10:
                    p10 += 1
                num_rel += 1
        except:
            continue

    ndcg1 = get_ndcg(sorted_res, rel_jud, 1)
    ndcg3 = get_ndcg(sorted_res, rel_jud, 3)
    ndcg5 = get_ndcg(sorted_res, rel_jud, 5)
    ndcg10 = get_ndcg(sorted_res, rel_jud, 10)

    return {"ndcg1": ndcg1, "ndcg3": ndcg3, "ndcg5": ndcg5, "ndcg10": ndcg10, "p5": p5/5.0,
            "p10": p10/10.0, "p1": p1/1.0, "p3": p3/3.0, "num_rel": num_rel}


def get_ndcg(sorted_res, rel_jud, cutoff):
    dcg = 0
    '''
    print (rel_jud.keys())
    for i in range(min(cutoff, len(sorted_res))):
        doc_id = sorted_res[i][0]
        if doc_id not in rel_jud.keys():
            rel_level = 0
        else: 
            rel_level = rel_jud[doc_id]
        print (doc_id, rel_level)
        dcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))
    '''
    for i in range(min(cutoff, len(sorted_res))):
        doc_id = sorted_res[i][0]
        rel_level = rel_jud[doc_id]
        dcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))

    ideal_sorted = {}
    for tup in sorted_res:
        try:
            ideal_sorted[tup[0]] = rel_jud[tup[0]]
        except:
            ideal_sorted[tup[0]] = 0
    ideal_sorted = sorted(ideal_sorted.iteritems(), key=operator.itemgetter(1), reverse=True)

    idcg = 0
    for i in range(min(cutoff, len(ideal_sorted))):
        doc_id = ideal_sorted[i][0]
        try:
            rel_level = rel_jud[doc_id]
        except:
            rel_level = 0
        idcg += (math.pow(2, rel_level) - 1) / (np.log2(i+2))
    if idcg == 0:
        idcg = 1

    return dcg/idcg


def evaluate(predictions, data_dir, output_dir):
    rel_jud = []
    qid_doc_dict = {}
    per_query_measures = {}
    avg_measure_dict = {"ndcg1": 0.0, "ndcg3": 0.0, "ndcg5": 0.0, "ndcg10": 0.0, "p1": 0.0, "p3": 0.0, "p5": 0.0,
                        "p10": 0.0, "num_rel": 0.0}
    measures_list = ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "p1", "p3", "p5", "p10", "num_rel"]

    lines = Utils.read_lines_from_file(data_dir)
    for i, line in enumerate(lines):
        rel_jud.append(float(line.split()[0]))
        qid = line.split()[1]
        qid_doc_dict[qid] = qid_doc_dict.get(qid, []) + [i]

    for qid in qid_doc_dict:
        res_dict = {}
        for doc_id in qid_doc_dict[qid]:
            res_dict[doc_id] = predictions[doc_id]
        sorted_res = sorted(res_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        per_query_measures[qid] = evaluate_res(sorted_res, rel_jud)

        for m in per_query_measures[qid]:
            avg_measure_dict[m] += per_query_measures[qid][m]

    for m in avg_measure_dict:
        avg_measure_dict[m] /= float(len(qid_doc_dict))

    with open(output_dir, 'w') as output_file:
        output_file.write("qid,ndcg1,ndcg3,ndcg5,ndcg10,p1,p3,p5,p10,num\n")
        for qid in per_query_measures.keys():
            output_file.write(qid)
            for m in measures_list:
                output_file.write("," + str(per_query_measures[qid][m]))
            output_file.write("\n")
        output_file.write("all")
        for m in measures_list:
            output_file.write("," + str(avg_measure_dict[m]))

    return per_query_measures, avg_measure_dict

def main():
    '''
    Inputdirectory = '../Data/'
    Outputdirectory = '../Models/LambdaMART/'
    
    with io.open(Inputdirectory + 'ClickData2.txt', 'r', encoding='utf-8') as infile:
        total_data = infile.readlines()
    train_data = total_data[:91369]
    test_data = total_data[91369:]
    '''
    '''
    delete_queries = [1087, 1090, 1106, 1112, 1132, 1145, 1163, 1197, 1216, 1217, 1225, 1229, 1232, 1240, 1246, 1286]
    filtered_test_data = []
    for t in test_data:
        qid = int(t.split(' ')[1].split(':')[1])
        if qid not in delete_queries:
            print (qid)
            filtered_test_data+=[t]
    '''
    #Utils.write_unicode_lines_to_file(Inputdirectory + 'ClickData2_train.txt', train_data)
    #Utils.write_unicode_lines_to_file(Inputdirectory + 'ClickData2_test.txt', test_data)
#    with io.open(Inputdirectory + 'ClickData2_train.txt', 'w', encoding='utf-8') as outfile:
#        outfile.write(train_data)
#    with io.open(Inputdirectory + 'ClickData2_test.txt', 'w', encoding='utf-8') as outfile:
#        outfile.write(test_data)
    #'../Data/New_features/Furniture/' + 'ClickData_New_features_with_BM25_Jan_1week.txt'
    '''
    train_data_dir = '../Models/search_log/Furniture/Data/' + 'ClickData_prob_features_four_components_Jan_2week_furniture.txt'
    test_data_dir = '../Models/search_log/Furniture/Data/' + 'ClickData_prob_features_four_components_Jan_4week_furniture.txt'
    #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
    model_dir = Outputdirectory + "search_log/Furniture/ClickData_prob_features_four_components_[1,2,3,5,6,8,9,11]_Jan_2week_furniture.model"
    performance_dir = Outputdirectory + "search_log/Furniture/ClickData_prob_features_four_components_[1,2,3,5,6,8,9,11]_Jan_4week_furniture.perf.txt"
    features_list = [str(i) for i in [1,2,3,5,6,8,9,11]]
    '''
    #2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39 - anchor and url features
    #41,42,43,44,45,46 - doc features
    '''
    InputDirectory = '../Web_data/MQ2007/fold1/'
    OutputDirectory = '../Results/LambdaMART/'


    train_data_dir = InputDirectory + 'train_division_1_fold123.txt'
    features = list(set(range(1,47)) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "train_division_1_fold123_wo_anchor_url.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['combined_test_set']: #'old_q_new_d', 'old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1',
        test_data_dir = InputDirectory + s + '.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_train_division_1_fold123_wo_anchor_url.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)
    '''

    OutputDirectory = '../Results/LambdaMART/'
    train_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_train_valid_division_1_LTR_IRF_features_wonr.txt'
    features = list(set(range(47,65)) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "tvd1_only_IRF_all_comb.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set']:
        test_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_tvd1_only_IRF_all_comb.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)


    OutputDirectory = '../Results/LambdaMART/'
    train_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_train_valid_division_1_LTR_IRF_features_wonr.txt'
    features = list(set(range(1,47)+[64]) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "tvd1_wo_anchor_url_only_IRF_comb.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set']:
        test_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_tvd1_wo_anchor_url_only_IRF_comb.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)


    OutputDirectory = '../Results/LambdaMART/'
    train_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_train_valid_division_1_LTR_IRF_features_wonr.txt'
    features = list(set(range(1,64)) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "tvd1_wo_anchor_url_only_IRF_all.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set']:
        test_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_tvd1_wo_anchor_url_only_IRF_all.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)

    OutputDirectory = '../Results/LambdaMART/'
    train_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_train_valid_division_1_LTR_IRF_features_wonr.txt'
    features = list(set(range(1,65)) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "tvd1_wo_anchor_url_only_IRF_all_comb.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set']:
        test_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_tvd1_wo_anchor_url_only_IRF_all_comb.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)


    OutputDirectory = '../Results/LambdaMART/'
    train_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_train_valid_division_1_LTR_IRF_features_wonr.txt'
    features = list(set(range(41,65)) - set([2,4,7,9,12,14,17,19,22,24,27,29,32,34,37,39]))
    print features
    features_list = [str(i) for i in features]
    model_dir = OutputDirectory + "tvd1_wo_content_IRF_all_comb.model"
    learn_model(train_data_dir, model_dir, features_list)    
    for s in ['old_q_old_d', 'new_q_old_d', 'new_q_new_doc_1', 'old_q_new_d','combined_test_set']:
        test_data_dir = '../Web_data/search_log_results/training_data/ClickData_ttd1_' + s + '_LTR_IRF_features_wonr.txt'
        #test_data_dir = Inputdirectory + 'Jan/ClickDataJan.txt'
        performance_dir = OutputDirectory + s + "_tvd1_wo_content_IRF_all_comb.perf.txt"        
        predictions = predict(model_dir, test_data_dir)
        #pickle.dump(predictions, open(Outputdirectory + 'search_log/LTR_New_features_w.p', 'wb'))
        evaluate(predictions, test_data_dir, performance_dir)

if __name__ == "__main__":
    main()
