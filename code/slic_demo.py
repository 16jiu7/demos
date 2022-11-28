#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:50:27 2022

@author: jiu7
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, find_contours
from skimage.util import img_as_float32, img_as_ubyte
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw

from utils import GraphedImage

import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skfmm
from easydict import EasyDict as edict

tmp_dir_for_demo = '/home/jiu7/Downloads/LadderNet-master/STARE_results/im0001.png'
whole_img = io.imread(tmp_dir_for_demo)
gt, pred = whole_img[605:605*2, :], whole_img[605*2:, :]
del whole_img
# devide pred into certain and uncertain part
graphedpred = GraphedImage(pred)
piece_list = graphedpred.piece_list
graphedpred.mark_some_nodes(to_draw_list = [1,2,3,7])
# In[]

threshold = threshold_otsu(pred)
certain_pred = img_as_float32(pred > threshold)
uncertain_pred = img_as_float32(np.clip(pred - certain_pred, a_min = 0, a_max = None))
#uncertain_pred = np.where(uncertain_pred > 0.01, 1, uncertain_pred)
io.imsave('certain_pred.png', img_as_ubyte(certain_pred))
io.imsave('uncertain_pred.png', img_as_ubyte(uncertain_pred))

# In[]
# slic on certain_pred

slic_label = slic(np.stack([certain_pred, certain_pred, certain_pred], axis = -1), 2000)
certain_pred_boundaries = mark_boundaries(certain_pred, slic_label, mode = 'inner',color = (0,0,1)) # blue
io.imsave('certain_slic_res.png', img_as_ubyte(certain_pred_boundaries))
uncertain_pred_boundaries = mark_boundaries(uncertain_pred, slic_label, mode = 'inner',color = (1,0,1))
io.imsave('uncertain_slic_res.png', img_as_ubyte(uncertain_pred_boundaries))

# register pieces
graph = nx.Graph()
slic_props = regionprops(slic_label, certain_pred)
piece_list = register_pieces(slic_label, certain_pred, uncertain_pred, slic_props, debug = True)

# mark all centers
label_list = [x for x in range(len(piece_list))]
kind0_list = kind1_list = kind2_list = []
for piece in piece_list:
    if piece.kind == 0:
        kind0_list.append(piece.label)
    elif piece.kind == 1:
        kind1_list.append(piece.label)
    else:
        kind2_list.append(piece.label)
draw_some_nodes(piece_list, kind0_list, certain_pred_boundaries, save = True)
# In[]
def draw_nodes(image, coords): # image is float btw 0,1
    for coord in coords:
        rr, cc = draw.circle_perimeter(coord[0], coord[1], 2) # radius = 2
        image[rr, cc] = (1,0,0) # color = red
    return image
    


# find edges

b2b_edge_search_threshold = 200 # big-node-to-big-node edge searching range 
b2m_edge_search_threshold = 100 # big-node-to-small-node edge searching range 
b2b_edge_connect_threshold = 0.1 # slic nodes farer than this won't be connected
# io.imsave('tt.png', tt)

node_list = list(graph.nodes)     

def get_dist(index, piece_cores, tt):
    sum_dist = 0
    for coord in piece_cores[index]:
        sum_dist += tt[coord[0], coord[1]]
    num_pixels = len(piece_cores[index])    
    return (sum_dist / num_pixels).astype(np.float32)

def get_neighboors(pred, i, narrow, piece_imgs, piece_labels, slic_label):
    try:
        phi = np.ones_like(pred)
        phi[piece_slices[i]] -= 2 * piece_imgs[i]

        tt = skfmm.travel_time(phi, speed = pred, narrow = narrow).astype(np.float32)    
        neighboor_mask = np.where(tt.filled(fill_value=0) > 0, 1, 0) 
        considered_nodes = list(np.unique(neighboor_mask * slic_label))
        #considered_nodes = list(set(considered_nodes) & (set(piece_labels)))
        considered_nodes.remove(node)
        considered_nodes.remove(0)
        
        assert len(considered_nodes) > 0, f'no neighboor were found for node {node}'
        print(f'{len(considered_nodes)} neighboors are selected for node {node}')
        print(sorted(considered_nodes))
        # be carefull that tt is a masked array
        tt = (tt - tt.min()) / (tt.max() - tt.min()) # normalization btw 0,1
        tt_filled = tt.filled(fill_value = 0).astype(np.float32)
        io.imsave(f'tt_for_node{node}.png', img_as_ubyte(tt))

        # neighboor nodes are checked if they have min geo dist to target node
        mean_geo_dists = []
        for _, each_neighboor in enumerate(considered_nodes):
            index = piece_labels.index(each_neighboor)
            mean_geo_dists.append(get_dist(index, piece_cores, tt_filled))  
        target = mean_geo_dists.index(min(mean_geo_dists))    
        
    except AssertionError:
        narrow += 50
        print(f'current narrow:{narrow}')
        get_neighboors(pred, i, narrow, piece_imgs, piece_labels, slic_label)
        
    return considered_nodes, target, min(mean_geo_dists)    

narrow = 200
for i, node in enumerate(node_list):
  
    considered_nodes, target, mean_geo_dist = get_neighboors(pred, i, narrow, piece_imgs, piece_labels, slic_label)
    graph.add_edge(node, considered_nodes[target], weight=b2b_edge_connect_threshold/(b2b_edge_connect_threshold+mean_geo_dist))
    print(f'edge constructed : {node}-{mean_geo_dist:.2f}-{considered_nodes[target]}\n')

    

# In[]
labels_to_draw = [89]
draw_some_nodes(piece_list, labels_to_draw, certain_pred_boundaries)


# In[]

visualize_graph(im = pred, graph = graph)













