import numpy as np
import jieba as jb
from gensim.models import Word2Vec, KeyedVectors

with open('./data/punctuations.txt') as f:
    PUNCTS = f.read().decode('utf-8').split()

jb.load_userdict('./data/dict.txt')

def load_data():
    data_paths = ['./data/atec_nlp_sim_train.csv', './data/atec_nlp_sim_train_add.csv']
    data = []
    for fpath in data_paths:
        with open(fpath) as f:
            for line in f:
                data.append(line.strip().split('\t')[1:])

    np.random.shuffle(data)
    return data

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

def word2vec(wordlevel=False):
    if wordlevel:
        wv_path = "./saved/wv_model_word.wv"
        corpus_path = './data/corpus_word.txt'
    else:
        wv_path = "./saved/wv_model_char.wv"
        corpus_path = './data/corpus_char.txt'

    try:
        word_vectors = KeyedVectors.load(wv_path, mmap='r')
    except IOError:
        sentences = []
        with open(corpus_path) as f:
            for line in f:
                sentences.append(line.decode('utf-8').split())
        wv_model = Word2Vec(size=300, min_count=2, sg=1)
        wv_model.build_vocab(sentences)
        wv_model.train(sentences, total_examples=wv_model.corpus_count, epochs=8)
        word_vectors = wv_model.wv
        word_vectors.save(wv_path)
    i2w = [u'<pad>', u'<unk>'] + word_vectors.index2entity
    vocab = dict(zip(i2w, range(len(i2w))))
    return vocab, word_vectors

def golve(wordlevel=False):
    if wordlevel:
        vocab_path = './data/vocab_word.txt'
        vectors_path = './data/vectors_word.txt'
    else:
        vocab_path = './data/vocab_char.txt'
        vectors_path = './data/vectors_char.txt'

    vocab = {u'<pad>':0, u'<unk>':1}
    with open(vocab_path) as f:
        for line in f:
            vocab[line.decode('utf-8').split()[0]] = len(vocab)

    word_vectors = {}
    with open(vectors_path) as f:
        for line in f:
            vals = line.decode('utf-8').split()
            word_vectors[vals[0]] = vals[1:]
    return vocab, word_vectors

class DataSet(object):
    """docstring for DataSet."""

    def __init__(self, maxlen=80, wordlevel=False, K=10, wv_model='GloVe'):
        super(DataSet, self).__init__()

        if wv_model=='GloVe':
            self.vocab, self.word_vectors = golve(wordlevel)
        else:
            self.vocab, self.word_vectors = word2vec(wordlevel)

        x1, x2, y = zip(*load_data())
        self.x1 = to_paded_seqs(x1, self.vocab, maxlen, wordlevel)
        self.x2 = to_paded_seqs(x2, self.vocab, maxlen, wordlevel)
        self.y = np.array(y, dtype=np.int16)

        fold_len = len(self.y) // K
        self.slices = []
        for i in range(K-1):
            self.slices.append(np.s_[i*fold_len:(i+1)*fold_len])
        self.slices.append(np.s_[(K-1)*fold_len:])

    def get_data(self, val_index=None):
        if val_index is None:
            return ([self.x1, self.x2], self.y), None

        slice = self.slices[val_index]
        x1_train = np.delete(self.x1, slice, axis=0)
        x2_train = np.delete(self.x2, slice, axis=0)
        y_train = np.delete(self.y, slice, axis=0)
        x1_val, x2_val, y_val = self.x1[slice], self.x2[slice], self.y[slice]
        return ([x1_train, x2_train], y_train), ([x1_val, x2_val], y_val)

    def random_split(self, val_size=0.1):
        n_samples = len(self.y)
        inds = np.arange(n_samples)
        np.random.shuffle(inds)
        val_inds = inds[:int(n_samples*val_size)]
        x1_train = np.delete(self.x1, val_inds, axis=0)
        x2_train = np.delete(self.x2, val_inds, axis=0)
        y_train = np.delete(self.y, val_inds, axis=0)
        x1_val, x2_val, y_val = self.x1[val_inds], self.x2[val_inds], self.y[val_inds]
        return ([x1_train, x2_train], y_train), ([x1_val, x2_val], y_val)

    def get_emb_matrix(self):
        emb_matrix = np.random.uniform(-0.01, 0.01, size=(len(self.vocab), 300))
        for w,i in self.vocab.items():
            if w in self.word_vectors:
                emb_matrix[i] = self.word_vectors[w]
        return emb_matrix.astype(np.float32)
