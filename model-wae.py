import numpy as np
import os

import datetime
import matplotlib.pyplot as plt

from utils import *
from WAE import *
from scipy import sparse

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

mainPath = 'C:\\Users\\nizhe\\Desktop\\python code\\ml-20m'
data_dir = '\\data'
unique_sid, n_items, train_data, vad_data_tr, vad_data_te = load_data(mainPath + data_dir)


N = train_data.shape[0]
idxlist = list(range(N))

n_epochs = 20
batch_size = 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))
batch_size_vad = 500
N_vad = vad_data_tr.shape[0]
idxlist_vad = list(range(N_vad))
total_anneal_steps = 200000
anneal_cap = 0.2

p_dims = [300, 600, n_items]

tf.reset_default_graph()
wae = WAE(p_dims, random_seed = 12345)
saver, logits_var, loss_var, train_op_var, merged_var = wae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype = tf.float64, shape = None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)

recall_var = tf.Variable(0.0)
recall_dist_var = tf.placeholder(dtype = tf.float64, shape = None)
recall_summary = tf.summary.scalar('recall_at_k_validation', recall_var)
recall_dist_summary = tf.summary.histogram('recall_at_k_hist_validation', recall_dist_var)
merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary, recall_summary, recall_dist_summary])


# In[ ]:


arch_str = "I-%s-I" % ('-'.join([str(d) for d in wae.dims[1:-1]]))

log_dir = mainPath + '\\log\\ml-20m\\wae\\{}'.format(arch_str) + str(datetime.datetime.today()).replace(':', '-').replace('.', '-')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
print("log directory: %s" % log_dir)

summary_writer = tf.summary.FileWriter(log_dir, graph = tf.get_default_graph())

ckpt_dir = mainPath + '\\chkpt\\ml-20m\\wae\\{}'.format(arch_str) + str(datetime.datetime.today()).replace(':', '-').replace('.', '-')
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)    
print("ckpt directory: %s" % ckpt_dir)


ndcgs_vad = []
recall_vad = []
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    best_ndcg = -np.inf
    
    update_count = 0
    
    for epoch in range(n_epochs):
        np.random.shuffle(idxlist)
        print (epoch)
        # train for one epoch
        print ('begin training...')
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = train_data[idxlist[st_idx : end_idx]]
            
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')           
            
            
            feed_dict = {wae.input_ph: X, 
                         wae.keep_prob_ph: 0.6, 
                         wae.batch_size : X.shape[0],
                         wae.anneal_ph: min(anneal_cap, 1. * update_count / total_anneal_steps)} 
            
            sess.run(train_op_var, feed_dict = feed_dict)

            if bnum % 100 == 0:
                try:
                    summary_train = sess.run(merged_var, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_train, global_step = epoch * batches_per_epoch + bnum) 
                except tf.errors.InvalidArgumentError:
                    pass
            
            update_count += 1
            
        print ('begin evaluating...')
        
        # compute validation NDCG
        ndcg_dist = []
        recall_dist = []
        
        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, N_vad)
            X = vad_data_tr[idxlist_vad[st_idx : end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
        
            pred_val = sess.run(logits_var, feed_dict={wae.input_ph : X} )
            
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            
            ndcg_tmp, recall_tmp = get_NDCG_Recall(pred_val, vad_data_te[idxlist_vad[st_idx : end_idx]], k_ndcg = 100, k_rcall = 50)
            
            ndcg_dist.append(ndcg_tmp)
            recall_dist.append(recall_tmp)
        
        ndcg_dist = np.concatenate(ndcg_dist)
        ndcg_ = ndcg_dist.mean()
        ndcgs_vad.append(ndcg_)
        
        recall_dist = np.concatenate(recall_dist)
        recall_ = recall_dist.mean()
        recall_vad.append(recall_)
        
        merged_valid_val = sess.run(merged_valid, feed_dict = {ndcg_var: ndcg_, ndcg_dist_var : ndcg_dist,
                                                               recall_var: recall_, recall_dist_var : recall_dist})
        summary_writer.add_summary(merged_valid_val, epoch)
        print (recall_)
        print (ndcg_)
        # update the best model (if necessary)
        if ndcg_ > best_ndcg:
            saver.save(sess, '{}/model'.format(ckpt_dir))
            best_ndcg = ndcg_


# In[ ]:

plt.figure(figsize = (12, 3))
plt.plot(ndcgs_vad)
plt.ylabel("Validation NDCG@100")
plt.xlabel("Epochs")


