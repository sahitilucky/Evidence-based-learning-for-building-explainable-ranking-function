import Utils
import random
import math
import numpy as np


def get_qid_list(data_dir):
    qid_list = []
    lines = Utils.read_lines_from_file(data_dir)
    for line in lines:
        qid = line.split()[1]
        if qid not in qid_list:
            qid_list.append(qid)
    return qid_list


# take the full data set in the ltr format, and generate two sets (for training and test)
def split_dataset(data_dir, training_portion, output_dir, train_ref_dir=None, test_ref_dir=None):
    data_dict = {}
    lines = Utils.read_lines_from_file(data_dir)
    for line in lines:
        qid = line.split()[1]
        data_dict[qid] = data_dict.get(qid, []) + [line]

    train_ids = []
    test_ids = []
    if train_ref_dir is not None:
        train_ids = get_qid_list(train_ref_dir)
        test_ids = get_qid_list(test_ref_dir)
    else:
        qid_list = data_dict.keys()
        random.shuffle(qid_list)
        num_training = int(math.floor(len(qid_list)*training_portion))
        train_ids = qid_list[0:num_training]
        test_ids = qid_list[num_training: len(qid_list)]

    train_lines = []
    test_lines = []

    for qid in train_ids:
        for line in data_dict[qid]:
            train_lines.append(line)

    for qid in test_ids:
        for line in data_dict[qid]:
            test_lines.append(line)

    Utils.write_lines_to_file(output_dir + ".test.dat", test_lines)
    Utils.write_lines_to_file(output_dir + ".train.dat", train_lines)


def get_data_stats(data_dir):
    data_types = ["test", "train", "vali"]
    query_dict = {}
    for i in range(len(data_types)):
        with open(data_dir + "/Fold2/" + data_types[i] + '.txt', 'r') as data_file:
            for line in data_file:
                rel_label = line.split(" ")[0]
                qid = line.split(" ")[1]
                query_dict[qid] = query_dict.get(qid, []) + [int(rel_label)]
    print "num queries: " + str(len(query_dict))
    num_doc = []
    num_rel = []

    for qid in query_dict:
        doc_counter = 0
        rel_counter = 0
        for label in query_dict[qid]:
            doc_counter += 1
            if label > 0:
                rel_counter += 1
        num_rel.append(rel_counter)
        num_doc.append(doc_counter)

    num_doc = np.asarray(num_doc)
    num_rel = np.asarray(num_rel)

    print "avg docs: " + str(np.mean(num_doc))
    print "std docs: " + str(np.std(num_doc))
    print "min docs: " + str(np.min(num_doc))
    print "max docs: " + str(np.max(num_doc))

    print "avg rel: " + str(np.mean(num_rel))
    print "std rel: " + str(np.std(num_rel))
    print "min rel: " + str(np.min(num_rel))
    print "num min rel: " + str(len([i for i in num_rel if i==0]))
    print "max rel: " + str(np.max(num_rel))


def main():
    train_ref_dir = "ClickData.train.dat"
    test_ref_dir = "ClickData.test.dat"
    data_dir = "/Users/saarkuzi/git/unbxd/Data/OrderData.txt"
    #split_dataset(data_dir, 0.7, "OrderData", train_ref_dir=train_ref_dir, test_ref_dir=test_ref_dir)

    get_data_stats("Data/MQ2007/")

if __name__ == '__main__':
    main()
