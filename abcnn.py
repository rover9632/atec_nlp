from __future__ import print_function

import tensorflow as tf
import numpy as np

"""
Implmenentaion of ABCNNs
(https://arxiv.org/pdf/1512.05193.pdf)
"""


class ABCNN():
    def __init__(self, s, w, d, v, em_mat=None, filters=64, l2_reg=1e-5, model_type=0, layers=1, lr=0.001):
        """
        :param s: the sentence length.
        :param w: the filter width.
        :param d: the dimensionality of word embedding.
        :param v: the size of vocabulary.
        :param em_mat: the word embedding matrix.
        :param filters: the number of filters.
        :param l2_reg: the L2 regularization coefficient.
        :param model_type: the type of the network(0:BCNN, 1:ABCNN1, 2:ABCNN2, 3:ABCNN3).
        :param layers: the number of convolution layers.
        """

        self.g = tf.Graph()
        with self.g.as_default():

            self.batch_x1 = tf.placeholder(tf.int32, [None, s], name='batch_x1')
            self.batch_x2 = tf.placeholder(tf.int32, [None, s], name='batch_x2')
            self.batch_y = tf.placeholder(tf.float32, [None], name='batch_y')

            x1 = self.embedding(self.batch_x1, em_mat, v=v, d=d)
            x2 = self.embedding(self.batch_x2, reuse=True)

            all_ap_01, all_ap_02 = self.all_pool(x1), self.all_pool(x2)
            contrasts = [] #[self.contrast(all_ap_01, all_ap_02)]

            if model_type == 1 or model_type == 3:
                x1, x2 = self.add_attention_feature_map(x1, x2, d, s)

            for i in range(layers):
                conv1 = self.conv(self.pad(x1, w), w, filters, l2_reg, i)
                conv2 = self.conv(self.pad(x2, w), w, filters, l2_reg, i, True)

                all_ap_1, all_ap_2 = self.all_pool(conv1), self.all_pool(conv2)
                contrasts.append(self.contrast(all_ap_1, all_ap_2))

                if i == (layers-1):
                    break

                x1, x2 = self.w_pool(conv1, conv2, w, model_type)
                #x1 = tf.layers.batch_normalization(x1)
                #x2 = tf.layers.batch_normalization(x2)

            x = tf.concat(contrasts, axis=-1)

            logits = tf.layers.dense(
                inputs=x,
                units=1,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            labels = tf.expand_dims(self.batch_y, axis=-1)

            self.predict_probs = tf.sigmoid(logits)
            self.predictions = tf.round(self.predict_probs)

            self.loss = tf.add(
                tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels, logits, 2)),
                tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            )

            self.optimizer = tf.train.AdamOptimizer(lr, name="optimizer").minimize(self.loss)


    def embedding(self, x, em_mat=None, reuse=False, v=None, d=None):
        with tf.variable_scope('embedding', reuse=reuse):
            if reuse:
                em_matrix = tf.get_variable('em_matrix')
            elif em_mat is None:
                assert v is not None, 'vocabulary size must not be None'
                assert d is not None, 'embedding dimensionality must not be None'
                em_matrix = tf.get_variable(
                    name='em_matrix',
                    dtype=tf.float32,
                    shape=[v, d],
                    initializer=tf.random_uniform_initializer(-0.05, 0.05)
                )
            else:
                em_matrix = tf.get_variable(
                    name='em_matrix',
                    dtype=tf.float32,
                    initializer=em_mat,
                    trainable=False
                )

            x = tf.nn.embedding_lookup(em_matrix, x)
            x = tf.matrix_transpose(x)
            return tf.expand_dims(x, axis=-1)


    def pad(self, x, w):
        return tf.pad(x, [[0,0], [0,0], [w-1,w-1], [0,0]])


    def conv(self, x, w, filters, l2_reg, layer, reuse=False):
        d = x.get_shape().as_list()[1]
        with tf.variable_scope('conv_'+str(layer), reuse=reuse):
            conved = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=[d, w],
                activation=tf.nn.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
            )
            conved_trans = tf.transpose(conved, [0, 3, 2, 1])
            return conved_trans


    def make_attention_matrix(self, x1, x2):
        squared = tf.square(tf.subtract(x1, tf.matrix_transpose(x2)))
        euclidean = tf.sqrt(tf.reduce_sum(squared, axis=1))
        return tf.divide(1.0, tf.add(1.0, euclidean))


    def add_attention_feature_map(self, x1, x2, d, s, l2_reg=1e-5):
        with tf.variable_scope("att_weights"):
            att_w = tf.get_variable(
                name="att_weights",
                shape=(d, s),
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
            )

            att_mat = self.make_attention_matrix(x1, x2)

            att_mat_T = tf.matrix_transpose(att_mat)
            x1_a = tf.expand_dims(tf.einsum("ij,kjl->kil", att_w, att_mat_T), -1)
            x2_a = tf.expand_dims(tf.einsum("ij,kjl->kil", att_w, att_mat), -1)

            x1 = tf.concat([x1, x1_a], axis=-1)
            x2 = tf.concat([x2, x2_a], axis=-1)
            return x1, x2


    def w_pool(self, x1, x2, w, model_type):
        if model_type == 2 or model_type == 3:
            att_mat = self.make_attention_matrix(x1, x2)
            att_1 = tf.expand_dims(tf.reduce_mean(att_mat, 2, True), axis=1)
            att_2 = tf.expand_dims(tf.reduce_mean(att_mat, 1, True), axis=-1)
            x1, x2 = tf.multiply(x1, att_1), tf.multiply(x2, att_2)

        x1 = tf.layers.average_pooling2d(x1, pool_size=[1, w], strides=1)
        x2 = tf.layers.average_pooling2d(x2, pool_size=[1, w], strides=1)
        return x1, x2


    def all_pool(self, x):
        s = x.get_shape().as_list()[2]
        all_ap = tf.layers.average_pooling2d(x, pool_size=[1, s], strides=1)
        return tf.squeeze(all_ap, [2, 3])


    def contrast(self, x1, x2):
        return tf.concat([tf.multiply(x1, x2), tf.abs(tf.subtract(x1, x2))], axis=-1)


    def f1_score(self, y_true, y_pred):
        tp = np.sum(np.logical_and(y_true, y_pred))
        pred_p = np.sum(y_pred) + 1e-6
        actual_p = np.sum(y_true) + 1e-6
        precision = tp / pred_p
        recall = tp / actual_p
        return (2 * precision * recall) / (precision + recall + 1e-6)


    def accuracy(self, y_true, y_pred):
        return np.mean(np.equal(y_true, y_pred))


    def fit(self, x, y, batch_size=32, epochs=5, validation_data=None):
        x1, x2 = x
        if validation_data is not None:
            [x1_val, x2_val], y_val = validation_data
            n_samples_val = len(y_val)

        with tf.Session(graph=self.g) as sess:
            sess.run(tf.global_variables_initializer())

            n_samples = len(y)
            for epoch in range(1, epochs+1):
                loss_train = []
                preds_train = np.empty((0), dtype=np.bool)

                i = 0
                while i < n_samples:
                    feed_dict = {
                        self.batch_x1: x1[i:i+batch_size],
                        self.batch_x2: x2[i:i+batch_size],
                        self.batch_y: y[i:i+batch_size]
                    }
                    _, loss, preds = sess.run(
                        fetches=[self.optimizer, self.loss, self.predictions],
                        feed_dict=feed_dict
                    )
                    loss_train.append(loss)
                    preds = preds.reshape(-1).astype(np.bool)
                    preds_train = np.append(preds_train, preds)
                    i += batch_size

                print('epoch {}:'.format(epoch))
                print(
                    '\ttrain: loss: {:.4f} , acc: {:.4f} , f1_score: {:.4f}'.format(
                        np.mean(loss_train),
                        self.accuracy(y, preds_train),
                        self.f1_score(y, preds_train)
                    )
                )

                if validation_data is not None:
                    loss_val = []
                    preds_val = np.empty((0), dtype=np.bool)
                    i = 0
                    while i < n_samples_val:
                        feed_dict = {
                            self.batch_x1: x1_val[i:i+batch_size],
                            self.batch_x2: x2_val[i:i+batch_size],
                            self.batch_y: y_val[i:i+batch_size]
                        }
                        loss, preds = sess.run(
                            fetches=[self.loss, self.predictions],
                            feed_dict=feed_dict
                        )
                        loss_val.append(loss)
                        preds = preds.reshape(-1).astype(np.bool)
                        preds_val = np.append(preds_val, preds)
                        i += batch_size

                    print(
                        '\tval: loss: {:.4f} , acc: {:.4f} , f1_score: {:.4f}'.format(
                            np.mean(loss_val),
                            self.accuracy(y_val, preds_val),
                            self.f1_score(y_val, preds_val)
                        )
                    )

            saver = tf.train.Saver()
            saver.save(sess, './saved/model')


    def predict(self, x1, x2, batch_size=32):
        with tf.Session(graph=self.g) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './saved/model')
            results = np.empty((0, 2), dtype=np.float32)
            i = 0
            while i < len(x1):
                feed_dict = {
                    self.batch_x1: x1[i:i+batch_size],
                    self.batch_x2: x2[i:i+batch_size]
                }
                probs, preds = sess.run(
                    fetches=[self.predict_probs, self.predictions],
                    feed_dict=feed_dict
                )
                result = np.concatenate((probs, preds), axis=-1)
                results = np.append(results, result, axis=0)
                i += batch_size
        return results
