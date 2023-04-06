#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, label
from skimage.util import img_as_ubyte, img_as_bool
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
from skimage.morphology import remove_small_objects
import skimage.io as io
import numpy as np
import networkx as nx
import torch
from data_handeler import RetinalDataset
import datetime
import networkx as nx

def get_bboxes_from_pred(pred, params = {'threshold': 'otsu', 'neg_iso_size': 32, 'n_slic_pieces': 800}):
    # larger neg_iso_size, less node, other params don't matter a lot
    assert pred.dtype == np.uint8
    if params['threshold'] == 'otsu':
        threshold = threshold_otsu(pred)
    certain_pred = pred >= threshold
    certain_pred = remove_small_objects(certain_pred, min_size = params['neg_iso_size'])
    
    all_bboxes = []
    iso_label = label(certain_pred)
    iso_regions = regionprops(iso_label)
    small_iso_bboxes = [x.bbox for x in iso_regions if max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17]
    all_bboxes += small_iso_bboxes
    
    large_isos = certain_pred.copy()
    for x in small_iso_bboxes: large_isos[x[0]:x[2], x[1]:x[3]] = 0
    slic_label = slic(large_isos, params['n_slic_pieces'], mask = large_isos, enforce_connectivity = False, compactness = 100)
    slic_regions = regionprops(slic_label)
    
    for x in slic_regions:
        if max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17:
            all_bboxes.append(x.bbox)
        else:
            offset = (x.bbox[0], x.bbox[1])
            patch = x.image
            n_pieces = ((patch.shape[0] // 17) + 1) * ((patch.shape[1] // 17) + 1)
            patch_slic = slic(patch, n_pieces, mask = patch, enforce_connectivity = False)
            patch_slic_regions = regionprops(patch_slic)
            patch_bboxes = [(region_patch.bbox[0] + offset[0], region_patch.bbox[1] + offset[1], region_patch.bbox[2] + offset[0], region_patch.bbox[3] + offset[1])\
                            for region_patch in patch_slic_regions]

            all_bboxes += patch_bboxes

    small_bboxes = [x for x in all_bboxes if (x[2] - x[0]) <= 17 and (x[3] - x[1]) <= 17]
    large_bboxes = [x for x in all_bboxes if (x[2] - x[0]) > 17 or (x[3] - x[1]) > 17]
    for idx, x in enumerate(small_bboxes): #expend small bboxes
        h, w = x[2] - x[0], x[3] - x[1]
        if h < 17 or w < 17:
            center = (int((x[2] + x[0]) / 2), int((x[3] + x[1]) / 2))
            small_bboxes[idx] = (center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8)
            
    for x in large_bboxes: #split large bboxes
        h, w = x[2] - x[0], x[3] - x[1]
        h_n, w_n = h // 17 + 1, w // 17 + 1
        sub_bboxes = []
        for i in range(h_n):
            for j in range(w_n):
                top_left = (x[0] + (i + 1) * 17, x[1] + (j + 1) * 17)
                sub_bboxes.append((top_left[0], top_left[1], top_left[0] + 17, top_left[1] + 17))
        small_bboxes += sub_bboxes
    # remove bboxes out of range    
    small_bboxes = list(filter(lambda x: (x[0]>=0) and (x[1]>=0) and(x[2]<=pred.shape[0]) and (x[3]<=pred.shape[1]), small_bboxes))
    small_bboxes = list(filter(lambda x: certain_pred[x[0]:x[2], x[1]:x[3]].sum() > 0, small_bboxes))
    return small_bboxes, certain_pred


def add_edges(bboxes, narrow = 100, n_links = 2):
    assert narrow < 256, 'param narrow is too large'
    centers = [(i, x[0] + 8, x[1] + 8) for i, x in enumerate(bboxes)]

    graph = nx.Graph()
    
    for i, x in enumerate(bboxes):
        graph.add_node(centers[i][0], center = (centers[i][1], centers[i][2]), bbox = x)
        
    for node in graph.nodes:
        node_x, node_y = graph.nodes[node]['center']
        neighbors = list(filter(lambda x: (abs(node_x - x[1]) < narrow) and (abs(node_y - x[2]) < narrow) and (node != x[0]), centers))

        X = np.array([neighbor[1] for neighbor in neighbors], dtype = np.uint16)
        Y = np.array([neighbor[2] for neighbor in neighbors], dtype = np.uint16)
        DST = (X - node_x) ** 2 + (Y - node_y) ** 2
        rank = np.argsort(DST)
        targets = [neighbors[idx][0] for idx in rank[:min(n_links, len(neighbors))]]

        edges = [(node, target) for target in targets] 
        graph.add_edges_from(edges)
    
    # connect isolated graph parts together
    # for isolate graph parts, each node is linked to n_link nearest nodes in main part
    N_components = nx.number_connected_components(graph)
    components = nx.connected_components(graph)
    components = sorted(components, key = len)
    _, small_componets = components[-1], components[:N_components - 1]
    all_nodes = set([i for i in range(len(graph.nodes))])
    
    for component in small_componets:
        other_nodes = all_nodes - component
        other_centers = [x for x in centers if x[0] in other_nodes]
        for node in component:
            node_x, node_y = graph.nodes[node]['center']
            neighbors = list(filter(lambda x: (abs(node_x - x[1]) < narrow) and (abs(node_y - x[2]) < narrow), other_centers))
            X = np.array([neighbor[1] for neighbor in neighbors], dtype = np.uint16)
            Y = np.array([neighbor[2] for neighbor in neighbors], dtype = np.uint16)
            DST = (X - node_x) ** 2 + (Y - node_y) ** 2
            rank = np.argsort(DST)
            targets = [neighbors[idx][0] for idx in rank[:min(1, len(neighbors))]]
            edges = [(node, target) for target in targets] 
            graph.add_edges_from(edges)  
        
    return graph    

def vis_bbox_graph(bboxes, graph, background, save_dir):
    for bbox in bboxes:
        start = (bbox[0], bbox[1])
        end = (bbox[2], bbox[3])
        rr,cc = draw.rectangle_perimeter(start, end)
        draw.set_color(background, (rr, cc), color = (255,0,0))
    for edge in graph.edges:
        start, end = edge
        r0, c0 = graph.nodes[start]['center']
        r1, c1 = graph.nodes[end]['center']
        rr,cc = draw.line(r0, c0, r1, c1)
        draw.set_color(background, (rr, cc), color = (0,0,255))
    io.imsave(save_dir, img_as_ubyte(background))

params_for_stare = {'threshold': 'otsu', 'neg_iso_size': 16, 'n_slic_pieces': 500}
params_for_drive = {'threshold': 'otsu', 'neg_iso_size': 16, 'n_slic_pieces': 500}

datas = RetinalDataset('STARE').all_data
for data in datas:
    pred = data.pred
    mask = data.fov_mask.astype(bool)
    pred = pred * mask
    bboxes, certain_pred = get_bboxes_from_pred(pred)
    graph = add_edges(bboxes)
    vis_bbox_graph(bboxes, graph, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
    print(graph)
    print(nx.number_connected_components(graph), '\n')        














