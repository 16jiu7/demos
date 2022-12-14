#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:17:23 2022

@author: jiu7
"""
from utils import GraphedImage
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

tmp_dir_for_demo = '/home/jiu7/Downloads/LadderNet-master/STARE_results/im0001.png'
mask = io.imread("/home/jiu7/4_retinal_datasets/STARE/masks/im0001.gif")
whole_img = io.imread(tmp_dir_for_demo)
ori, gt, pred = whole_img[:605, :], whole_img[605:605*2, :], whole_img[605*2:, :]
del whole_img

graphedpred = GraphedImage(ori, pred, mask, 3000) # this number of pieces as input != actual number of pieces
piece_list = graphedpred.piece_list

a = graphedpred.graph
print(a)
graphedpred.draw_graph()
print(f'the graph has {nx.number_connected_components(a)} components')
# In[]
slic_label = graphedpred.slic_label

# In[]
uncertain = graphedpred.uncertain_pred
threshold = threshold_otsu(uncertain)
print(threshold)
enhanced_uncertain = np.where(uncertain > threshold, 1, 0)
io.imsave('enhanced_uncertain.png', enhanced_uncertain)








