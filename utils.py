# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:18:56 2019

@author: nizhe
"""

import numpy as np
import pandas as pd
import os
import bottleneck as bn
from scipy import sparse

# In[4]:


def load_data(data_dir):
    
    unique_sid = list()
    with open(os.path.join(data_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    n_items = len(unique_sid)
    
    tp = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    tp_tr = pd.read_csv(os.path.join(data_dir, 'vad_tr.csv'))
    tp_te = pd.read_csv(os.path.join(data_dir, 'vad_te.csv'))
    
    n_users = tp['uid'].max() + 1
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    
    rows, cols = tp['uid'], tp['sid']
    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']
    
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype = 'float64', shape=(n_users, n_items))
    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype = 'float64', shape = (end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype = 'float64', shape = (end_idx - start_idx + 1, n_items))
    
    return unique_sid, n_items, data, data_tr, data_te


def get_NDCG_Recall(X_pred, heldout_batch, k_ndcg = 100, k_rcall = 100):
    
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k_ndcg, axis = 1)
    topk_part = X_pred[np.transpose(np.array([np.arange(batch_users)])), idx[:, :k_ndcg]]
    idx_part = np.argsort(-topk_part, axis = 1)
    idx_topk = idx[np.transpose(np.array([np.arange(batch_users)])), idx_part]
    tp = 1. / np.log2(np.arange(2, k_ndcg + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis = 1)
    IDCG = np.array([(tp[:min(n, k_ndcg)]).sum() for n in heldout_batch.getnnz(axis = 1)])
    
    X_pred_binary = np.zeros_like(X_pred, dtype = bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k_rcall]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis = 1)).astype(np.float32)
    recall = tmp / np.minimum(k_rcall, X_true_binary.sum(axis = 1))
    
    return DCG / IDCG, recall


