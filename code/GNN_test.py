#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:17:23 2022

@author: jiu7
"""
from make_graph import GraphedImage
import os, sys
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, find_contours
from skimage.util import img_as_float32, img_as_ubyte
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import datetime
from handcrafted_feats import hand_feats_extractor
from data_handeler import RetinalDataset
import models.GAT
from scipy.stats import shapiro, mannwhitneyu, levene, ttest_ind, normaltest, zscore
# In[]

tmp_dir_for_demo = '/home/jiu7/Downloads/LadderNet-master/STARE_results/im0001.png'
mask = io.imread("/home/jiu7/4_retinal_datasets/STARE/masks/im0001.gif")
whole_img = io.imread(tmp_dir_for_demo)
_, gt, pred = whole_img[:605, :], whole_img[605:605*2, :], whole_img[605*2:, :]
ori = io.imread('/home/jiu7/4_retinal_datasets/STARE/images/im0001.ppm')

starttime = datetime.datetime.now()
graphedpred = GraphedImage(ori, pred, mask, 3000) # this number of pieces as input != actual number of pieces
endtime = datetime.datetime.now()
print(f'Run time : {endtime - starttime}s')

piece_list = graphedpred.piece_list
graphedpred.draw_graph()
a = graphedpred.graph
print(f'the graph has {nx.number_connected_components(a)} components\n')

label_slices = [(label, a.nodes[label]['slices']) for label in a.nodes]
starttime = datetime.datetime.now()
print('extracting hand-crafted node features')
feats = hand_feats_extractor(img = ori, label_slices = label_slices, label_img = graphedpred.slic_label, intensity_img = pred)
feats = zscore(feats, axis = 0, ddof = 1, nan_policy = 'raise')
endtime = datetime.datetime.now()
print(f'Run time : {endtime - starttime}s')    

# In[]

drive = RetinalDataset('DRIVE')
drive_train = drive.all_training

'''
input for GAT: 
1.node feats: (N_nodes, N_feats)
2.edges: (2, N_edges)
'''



































