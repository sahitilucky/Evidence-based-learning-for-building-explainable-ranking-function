import re
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def read_lines_from_file(input_dir):
    lines = []
    with open(input_dir, 'r') as input_file:
        for line in input_file:
            lines += [line.rstrip('\n')]
    return lines


def write_unicode_lines_to_file(output_dir, lines):
    output = ''
    with open(output_dir, 'w') as output_file:
        for line in lines:
            #print line
            output += line.encode('utf8')
        output = output.rstrip('\n')
        output_file.write(output)


def write_lines_to_file(output_dir, lines):
    output = ''
    with open(output_dir, 'w') as output_file:
        for line in lines:
            output += str(line) + '\n'
        output = output.rstrip('\n')
        output_file.write(output)


def output_dict_to_csv(data_dict_list, output_dir, separator=','):

    output_str = ''
    with open(output_dir, 'w+') as output_file:
        keys_list = []
        for key in data_dict_list[0].keys():
            output_str += key + separator
            keys_list += [key]
        output_str = output_str.rstrip(separator) + '\n'

        for single_dict in data_dict_list:
            for key in keys_list:
                output_str += str(single_dict[key]) +separator
            output_str = output_str.rstrip(separator) + '\n'
        output_str = output_str.rstrip('\n')
        output_file.write(output_str)


def l1_norm(feat_vec):
    norm_factor = sum(feat_vec.values())
    for key in feat_vec.keys():
        feat_vec[key] = float(feat_vec[key]) / norm_factor
    return feat_vec


def merge_dicts(dicts_list, coefficients):
    result = dict.fromkeys(dicts_list[0].keys(), 0)
    for key in result.keys():
        for i, single_dict in enumerate(dicts_list):
            result[key] += coefficients[i] * single_dict[key]
    return result


def clean_xml_tags(text, separator=''):
    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    return text


def remove_list_element(input_list, indexes):
    new_list = []
    for i, element in enumerate(input_list):
        if i in indexes:
            continue
        new_list += [input_list[i]]
    return new_list


def remove_xml_tag(tag_name, text):
    text = text.split('<' + tag_name)
    modified_text = text[0]
    for i in range(1, len(text)):
        modified_text += text[i].split('</' + tag_name + '>')[1]
    return modified_text


def get_sparse_vecs(sparse_matrix, filter_list=None):
    data = sparse_matrix.data
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    if filter_list is None:
        filter_list = []

    mat = []
    for row_id in range(len(indptr)-1):
        vec = []
        for i in range(indptr[row_id], indptr[row_id+1]):
            col_id = indices[i]
            point = data[i]
            if point not in filter_list:
                vec += [(col_id, point)]
        mat += [vec]
    return mat


def create_sparse_martix(vecs, shape):
    rows = []
    cols = []
    data = []

    for i in range(len(vecs)):
        for tup in vecs[i]:
            col = tup[0]
            point = tup[1]
            rows += [i]
            cols += [col]
            data += [point]

    return csr_matrix((data, (rows, cols)), shape=shape).toarray()

def main():
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    a = csr_matrix((data, indices, indptr), shape=(3, 3))

if __name__ == "__main__":
    main()