#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:34:53 2023
measure how useful each hand-crafted feature is
experiment on DRIVE
@author: jiu7
"""
# In[preparing for data]
from handcrafted_feats import hand_feats_extractor
from data_handeler import RetinalDataset
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
import random
from random import sample, shuffle
import os
from scipy.stats import shapiro, mannwhitneyu, levene, ttest_ind

N_SLIC_EACH = 2000
N_POS = N_NEG = 10000
SEED = 10
random.seed(SEED)
drive = RetinalDataset('DRIVE')
drive_train = drive.all_training

if not os.path.isfile('feats_vessel.npy'):
    feats = []
    if_vessel = []
    for i, data in enumerate(drive_train):
        slic_label = slic(data.ori, n_segments = N_SLIC_EACH, mask = data.fov_mask)
        props = regionprops(slic_label)
        slices = [prop.slice for prop in props]
        label_slices = [(1, slice_i) for slice_i in slices]
        flags = [np.sum(data.gt[prop.slice]) > 0 for prop in props]
        feat = hand_feats_extractor(data.ori, label_slices)
        print(f'img {i + 1} finished')
        feats.append(feat)
        if_vessel += flags
    feats = np.concatenate(feats, axis = 0) # shape: N_superpixels * N_feats
    feats_vessel = {'feats': feats, 'if_vessel': if_vessel}
    np.save('feats_vessel', feats_vessel, allow_pickle = True)    
else:
    feats_vessel = np.load('feats_vessel.npy', allow_pickle = True).item()  
    feats = feats_vessel['feats']
    if_vessel = feats_vessel['if_vessel']

positive_idxs = [i for i, x in enumerate(if_vessel) if x]
negative_idxs = [i for i, x in enumerate(if_vessel) if not x]

shuffle(positive_idxs)
shuffle(negative_idxs)
# indexs for the experiment
positives = positive_idxs[:N_POS]
negatives = negative_idxs[:N_POS]
del positive_idxs, negative_idxs, if_vessel, N_NEG, N_POS, N_SLIC_EACH, drive, drive_train, feats_vessel

# In[starting the experiment]


# remove all-identical feats since normaltest can not handel this
N_feats = feats.shape[-1]
useful_feats_idx = [x for x in range(N_feats)]
print(f'{N_feats} feats in total')
for feat_idx in range(N_feats):
    feat = feats[:, feat_idx]
    if feat.std() == 0:
        print(f'feat {feat_idx} is useless')
        useful_feats_idx.remove(feat_idx)
pass
