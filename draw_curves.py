# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:40:02 2019

@author: nizhe
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ndcg_ep20 = pd.read_csv('ndcg_ep20.csv', header = None).values
ndcg_ep100 = pd.read_csv('ndcg_ep100.csv', header = None).values

recall_ep20 = pd.read_csv('recall_ep20.csv', header = None).values
recall_ep100 = pd.read_csv('recall_ep100.csv', header = None).values
recall_ep100_sample20 = np.zeros(20)
for i in range(20):
    recall_ep100_sample20[i] = recall_ep100[i * 5]

ndcg_ep20_vae = pd.read_csv('ndcg_ep20_vae.csv', header = None).values
recall_ep20_vae = pd.read_csv('recall_ep20_vae.csv', header = None).values

plt.figure()
plt.xticks(np.arange(0, 20))
plt.plot(np.arange(20), ndcg_ep20)
plt.plot(np.arange(20), ndcg_ep20_vae)
plt.legend(['WAE', 'VAE'])
plt.xlabel("Epochs")
plt.ylabel("NDCG@100 in 20 epoches")
#
plt.figure()
plt.xticks(np.arange(0, 20))
plt.plot(np.arange(20), recall_ep100_sample20)
plt.plot(np.arange(20), recall_ep20_vae)
plt.legend(['WAE', 'VAE'])
plt.xlabel("Epochs")
plt.ylabel("Recall@100 in 20 epoches")