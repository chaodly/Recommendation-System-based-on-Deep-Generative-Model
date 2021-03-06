# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:00:22 2019

@author: nizhe
"""

import tensorflow as tf

class GAN(object):
    
    def __init__(self, real_size, fake_size, g_units, d_units, alpha, lr, smooth):
        
        self.real_size = real_size
        self.fake_size = fake_size
        self.g_units = g_units
        self.d_units = d_units
        self.alpha = alpha
        self.lr = lr
        self.smooth = smooth
        
    def get_inputs(self):
    
        real = tf.placeholder(tf.float32, [None, self.real_size], name='real_matrix')
        fake = tf.placeholder(tf.float32, [None, self.fake_size], name='fake_matrix')
        
        return real, fake   
        
    def generator(self, fake, out_dim, rate = 0.2, reuse = False):
        
        with tf.variable_scope("generator", reuse = reuse):
            hidden0 = tf.layers.dense(fake, self.g_units)
            hidden1 = tf.maximum(self.alpha * hidden0, hidden0)
            hidden2 = tf.layers.dropout(hidden1, rate)
            logits = tf.layers.dense(hidden2, out_dim)
            outputs = tf.tanh(logits)
            
            return logits, outputs
            

    def discriminator(self, real, reuse = False):
        
        with tf.variable_scope("discriminator", reuse = reuse):
            hidden0 = tf.layers.dense(real, self.d_units)
            hidden1 = tf.maximum(self.alpha * hidden0, hidden0)
            logits = tf.layers.dense(hidden1, 1)
            outputs = tf.sigmoid(logits)
            
            return logits, outputs
    
    def build_graph(self):
        
        real, fake = self.get_inputs()
        g_logits, g_outputs = self.generator(fake, self.real_size)
        d_logits_real, d_outputs_real = self.discriminator(real)
        d_logits_fake, d_outputs_fake = self.discriminator(g_outputs, reuse = True)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                        logits = d_logits_real, labels = tf.ones_like(d_logits_real)) * (1 - self.smooth))
        
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                        logits = d_logits_fake, labels = tf.zeros_like(d_logits_fake)))
        
        d_loss = tf.add(d_loss_real, d_loss_fake)
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                    logits = d_logits_fake, labels = tf.ones_like(d_logits_fake)) * (1 - self.smooth))

        
        train_vars = tf.trainable_variables()
        
        g_vars = [var for var in train_vars if var.name.startswith("generator")]
        d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
        
        # optimizer
        d_train_opt = tf.train.AdamOptimizer(self.lr).minimize(d_loss, var_list = d_vars)
        g_train_opt = tf.train.AdamOptimizer(self.lr).minimize(g_loss, var_list = g_vars)
        
        saver = tf.train.Saver(var_list = g_vars)
        
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        merged = tf.summary.merge_all()
        
        return d_train_opt, g_train_opt, merged, saver