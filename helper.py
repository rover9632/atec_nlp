#coding=utf-8

import numpy as np
import jieba as jb
import re
import random
import sys

def train_test_split(fnames, test_size=0.1):
    x = []
    for fname in fnames:
        with open(fname) as f:
            for line in f:
                x.append(line.split('\t')[1:])

    random.shuffle(x)
    p = int(len(x) * test_size)

    for data,fname in zip([x[:-p], x[-p:]], ['./train.csv', './validation.csv']):
        with open(fname, 'w') as f:
            i = 1
            for item in data:
                f.write('\t'.join([str(i)]+item))
                i += 1
    pass

def data2corpus(data, corpus):
    with open('./punctuation.txt') as f:
        punctuation = f.read().decode('utf-8').split()

    with open(data) as fi, open(corpus, 'w') as fo:
        for line in fi:
            line = re.sub(u'[。？！.?!]', u'\n', line.decode('utf-8'))
            _, sen1, sen2, _ = line.split(u'\t')
            sen1 = [w for w in sen1 if w not in punctuation]
            sen2 = [w for w in sen2 if w not in punctuation]
            fo.write((u' '.join(sen1) + u'\n').encode('utf-8'))
            fo.write((u' '.join(sen2) + u'\n').encode('utf-8'))
    pass

if __name__ == '__main__':
    #train_test_split(sys.argv[1:])
    data2corpus('./train.csv', './corpus.txt')
