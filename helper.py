#coding=utf-8

import numpy as np
import jieba as jb
import re
import random
import sys

# this function is not needed for now
def train_test_split(in_data_paths, test_size=0.1):
    x = []
    for fpath in in_data_paths:
        with open(fpath) as f:
            for line in f:
                x.append(line.split('\t')[1:])

    random.shuffle(x)
    p = int(len(x) * test_size)
    out_data_paths = ['./data/train.csv', './data/validation.csv']

    for data,fpath in zip([x[:-p], x[-p:]], out_data_paths):
        with open(fpath, 'w') as f:
            i = 1
            for item in data:
                f.write('\t'.join([str(i)]+item))
                i += 1
    pass

def load_data(data_paths):
    data = []
    for fpath in data_paths:
        with open(fpath) as f:
            for line in f:
                data.append(line)
    return data

def data2corpus(data, corpus_path, wordlevel=False):
    with open('./data/punctuations.txt') as f:
        punctuations = f.read().decode('utf-8').split()

    if wordlevel:
        jb.load_userdict('./data/dict.txt')

    with open(corpus_path, 'w') as fo:
        for line in data:
            line = re.sub(u'[。？！.?!]', u'\n', line.decode('utf-8'))
            _, sen1, sen2, _ = line.split(u'\t')
            if wordlevel:
                sen1, sen2 = jb.lcut(sen1), jb.lcut(sen2)
            sen1 = [w for w in sen1 if w not in punctuations]
            sen2 = [w for w in sen2 if w not in punctuations]
            fo.write((u' '.join(sen1) + u'\n').encode('utf-8'))
            fo.write((u' '.join(sen2) + u'\n').encode('utf-8'))
    pass


if __name__ == '__main__':
    #train_test_split(sys.argv[1:])
    data_paths = ['./data/atec_nlp_sim_train.csv', './data/atec_nlp_sim_train_add.csv']
    data = load_data(data_paths)
    data2corpus(data, './data/corpus_char.txt')
    data2corpus(data, './data/corpus_word.txt', True)
