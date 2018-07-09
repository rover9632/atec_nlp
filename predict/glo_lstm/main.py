#/usr/bin/env python
#coding=utf-8
import sys
from keras.models import load_model
import keras.backend as K
import jieba as jb
import os
import numpy as np

MAX_LEN = 80
EMD_DIM = 300

with open('./punctuations.txt') as f:
    PUNCTS = f.read().decode('utf-8').split()

def tokenize(texts, wordlevel=False):
    sens = []
    for sen in texts:
        if wordlevel:
            words = jb.lcut(sen)
        else:
            words = list(sen.decode('utf-8'))
        words = [w for w in words if w.strip() and w not in PUNCTS]
        sens.append(words)
    return sens

def to_int_seqs(seqs, vocab):
    rs = []
    for s in seqs:
        rs.append([vocab[w] if w in vocab else vocab[u'<unk>'] for w in s])
    return rs

def pad_sequences(sequences, maxlen, value=0):
    x = np.empty((len(sequences), maxlen), dtype=np.int32)
    x.fill(value)
    for i, seq in enumerate(sequences):
        if len(seq) >= maxlen:
            x[i] = seq[:maxlen]
        else:
            x[i,:len(seq)] = seq
    return x

def to_paded_seqs(x, vocab, maxlen, wordlevel=False):
    return pad_sequences(to_int_seqs(tokenize(x, wordlevel), vocab), maxlen=maxlen)

def fscore(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred)
    pred_p = K.sum(y_pred) + K.epsilon()
    actual_p = K.sum(y_true) + K.epsilon()
    precision = tp / pred_p
    recall = tp / actual_p
    return (2 * precision * recall) / (precision + recall + K.epsilon())

def weighted_binary_crossentropy(y_true, y_pred):
    p = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    loss = -(1.4 * y_true * K.log(p) + 0.6 * (1. - y_true) * K.log(1. - p))
    return K.mean(loss)

def predict(x, model_path):
    model = load_model(
        model_path,
        custom_objects={
            'fscore':fscore,
            'weighted_binary_crossentropy': weighted_binary_crossentropy
        }
    )
    preds = model.predict(x, batch_size=128)
    return preds

def process(inpath, outpath):
    vocab = {u'<pad>':0, u'<unk>':1}
    with open('./vocab.txt') as f:
        for line in f:
            vocab[line.decode('utf-8').split()[0]] = len(vocab)

    linenos, sens1, sens2 = [], [], []
    with open(inpath, 'r') as f:
        for line in f:
            lineno, sen1, sen2 = line.strip().split('\t')
            linenos.append(lineno)
            sens1.append(sen1)
            sens2.append(sen2)

    sens1, sens2 = to_paded_seqs(sens1, vocab, MAX_LEN), to_paded_seqs(sens2, vocab, MAX_LEN)
    model_path = './models'
    predses = []
    for fname in os.listdir(model_path):
        preds = predict([sens1, sens2], os.path.join(model_path, fname))        
        predses.append(preds)

    final_preds = np.mean(np.concatenate(predses, axis=-1), axis=-1)
    
    with open(outpath, 'w') as f:
        for lineno, pred in zip(linenos, final_preds):
            if pred >= 0.5:
                f.write(lineno + '\t1\n')
            else:
                f.write(lineno + '\t0\n')

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
