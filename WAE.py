# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:41:20 2019

@author: nizhe
"""
import tensorflow as tf

class WAE(object):
    
    def __init__(self, p_dims, lr = 0.004, random_seed = None):
        
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]
        self.dims = self.q_dims + self.p_dims[1:]
        self.lr = lr
        self.random_seed = random_seed
        self.construct_placeholders()

    def construct_placeholders(self):
        
        self.input_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape = None)
        self.is_training_ph = tf.placeholder_with_default(0., shape = None)
        self.batch_size = tf.placeholder_with_default(500, shape = None)
        self.anneal_ph = tf.placeholder_with_default(1., shape = None)
        
    def encoder(self):
        
        h = tf.nn.dropout(tf.nn.l2_normalize(self.input_ph, 1), self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]
                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis = 1))
        
        return h, mu_q, std_q, KL

    def decoder(self, z):
        
        h = z
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b
            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h
    

    def construct_weights(self):
        
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            self.weights_q.append(tf.get_variable(name = weight_key, shape = [d_in, d_out],
                initializer = tf.contrib.layers.xavier_initializer(seed = self.random_seed)))
            self.biases_q.append(tf.get_variable(name = bias_key, shape =[d_out],
                initializer = tf.truncated_normal_initializer(stddev = 0.001, seed = self.random_seed)))
            
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(name = weight_key, shape =[d_in, d_out],
                initializer = tf.contrib.layers.xavier_initializer(seed = self.random_seed)))
            self.biases_p.append(tf.get_variable(name = bias_key, shape =[d_out],
                initializer = tf.truncated_normal_initializer(stddev = 0.001, seed=self.random_seed)))
            
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])
            
    def mmd_loss(self, X, Y):
    
        p2_norm_x = tf.reduce_sum(tf.pow(X, 2), axis = 1)
        p2_norm_x = tf.reshape(p2_norm_x, [1, self.batch_size])
        norm_x = tf.reduce_sum(X, axis = 1)
        norm_x = tf.reshape(norm_x, [1, self.batch_size])
        prod_x = tf.matmul(norm_x, norm_x, transpose_b = True)
        dists_x = p2_norm_x + tf.transpose(p2_norm_x) - 2 * prod_x
        
        p2_norm_y = tf.reduce_sum(tf.pow(Y, 2), axis = 1)
        p2_norm_y = tf.reshape(p2_norm_y, [1, self.batch_size])
        norm_y = tf.reduce_sum(Y, axis = 1)
        norm_y = tf.reshape(norm_y, [1, self.batch_size])
        prod_y = tf.matmul(norm_y, norm_y, transpose_b = True)
        dists_y = p2_norm_y + tf.transpose(p2_norm_y) - 2 * prod_y
        
        dot_prd = tf.matmul(norm_x, norm_y, transpose_b = True)
        dists_c = p2_norm_x + tf.transpose(p2_norm_y) - 2 * dot_prd
        
        stats = 0.0
        
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = 2 * self.p_dims[0] * scale
            res = C / (C + dists_x) + C / (C + dists_y)
            res1 = (1 - tf.eye(self.batch_size)) * res
            res1 = tf.reduce_sum(tf.reduce_sum(res1, axis = 0), axis = 0) / tf.cast((self.batch_size - 1), tf.float32) / tf.cast(self.batch_size, tf.float32)
            res2 =  C / (C + dists_c)
            res2 = tf.reduce_sum(tf.reduce_sum(res2, axis = 0), axis = 0) * 2/ tf.cast(self.batch_size, tf.float32) / tf.cast(self.batch_size, tf.float32)
            stats += res1 - res2
    
        return stats
    
    def build_graph(self):
        
        self.construct_weights()
        z_real, mu_q, std_q, KL = self.encoder()
        z_fake = mu_q + tf.random_normal(tf.shape(std_q)) * std_q
        logits = self.decoder(z_fake)
        
        log_softmax_var = tf.nn.log_softmax(logits)
        
        saver = tf.train.Saver()

        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph, axis = -1))
        
        mmd_loss = self.mmd_loss(z_real, z_fake)
        
        reg = tf.contrib.layers.l2_regularizer(0.01)
        reg_var = tf.contrib.layers.apply_regularization(reg, self.weights_q + self.weights_p)
        
        total_loss = neg_ll - mmd_loss
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(total_loss)

        tf.summary.scalar('mmd_loss', mmd_loss)
        tf.summary.scalar('total_loss', total_loss)
        merged = tf.summary.merge_all()

        return saver, logits, total_loss, train_op, merged