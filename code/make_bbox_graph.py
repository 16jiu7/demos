#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:13:20 2023

@author: jiu7
"""
import os, pickle, sys
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, label
from skimage.util import img_as_ubyte, img_as_bool, img_as_float32
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
from skimage.morphology import remove_small_objects, thin
from skimage.exposure import equalize_adapthist
from skimage.transform import probabilistic_hough_line
import skimage.io as io
import numpy as np
import networkx as nx
from data_handeler import RetinalDataset
import random
import matplotlib.pyplot as plt
from datetime import datetime
from bwmorph import bwmorph
from skimage.transform import resize

def get_bboxes_from_pred(pred, n_bboxes = 1000, params = {'threshold': 'otsu', 'neg_iso_size': 6, 'n_slic_pieces': 700,\
                                                         'remove_keep': 36}):
    # larger neg_iso_size, less node, other params don't matter a lot
    if pred.dtype != np.uint8: pred = img_as_ubyte(pred)
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
            small_bboxes[idx] = (center[0] - 8, center[1] - 8, center[0] + 9, center[1] + 9)
            
    for x in large_bboxes: #split large bboxes
        h, w = x[2] - x[0], x[3] - x[1]
        h_n, w_n = h // 17 + 1, w // 17 + 1
        sub_bboxes = []
        for i in range(h_n):
            for j in range(w_n):
                top_left = (x[0] + (i + 1) * 17, x[1] + (j + 1) * 17)
                sub_bboxes.append((top_left[0], top_left[1], top_left[0] + 18, top_left[1] + 18))
        small_bboxes += sub_bboxes
    # remove bboxes out of range    
    small_bboxes = list(filter(lambda x: (x[0]>=0) and (x[1]>=0) and(x[2]<=pred.shape[0]) and (x[3]<=pred.shape[1]), small_bboxes))
    small_bboxes = list(filter(lambda x: certain_pred[x[0]:x[2], x[1]:x[3]].sum() > 0, small_bboxes))
    
    # add random bboxes if not enough
    n_to_add = n_bboxes - len(small_bboxes)
    if n_to_add >= 0:
        for i in range(n_to_add):
            h, w = pred.shape[0], pred.shape[1]
            r_point = (random.randint(0, h - 18), random.randint(0, w - 18))
            r_bbox = (r_point[0], r_point[1], r_point[0] + 18, r_point[1] + 18)
            small_bboxes.append(r_bbox)  
    else:
        random.shuffle(small_bboxes)
        small_bboxes = small_bboxes[:n_bboxes]     
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
        
    return nx.DiGraph(graph)    

def vis_bbox_graph(graph, background):
    #vis_bbox_graph(bboxes, graph, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
    if background.ndim == 2:
        background = np.stack([background]*3, axis = -1)
    if background.dtype == np.float32: background = img_as_ubyte(background)    
    for node in graph.nodes:
        cpoint = graph.nodes[node]['center']
        start = (cpoint[0] - 8, cpoint[1] - 8)
        end = (cpoint[0] + 8, cpoint[1] + 8)
        rr,cc = draw.rectangle_perimeter(start, end)
        draw.set_color(background, (rr, cc), color = (255,0,0))
    for edge in graph.edges:
        start, end = edge
        r0, c0 = graph.nodes[start]['center']
        r1, c1 = graph.nodes[end]['center']
        rr,cc = draw.line(r0, c0, r1, c1)
        draw.set_color(background, (rr, cc), color = (0,0,255))
    return img_as_ubyte(background)


def vis_lines(background, lines):
    if background.dtype == np.float32: background = img_as_ubyte(background)
    if background.ndim == 2: background = np.stack([background] * 3, axis = -1)
    if not isinstance(lines, list): lines = [lines]
    for line in lines:
        r0, c0, r1, c1 = line[0][0], line[0][1], line[1][0], line[1][1]
        rr, cc = draw.line(r0,c0,r1,c1)
        draw.set_color(background, (rr, cc), color = (0,0,255))
        
    points = []
    for line in lines: points.append(line[0]), points.append(line[1])
    points = sorted(points, key = lambda x : x[0])
    for point in points:
        rr, cc = draw.disk(point, radius = 2)
        draw.set_color(background, (rr, cc), color = (255, 0, 0))    
    return background    

def vis_points(background, points):
    if background.dtype == np.float32: background = img_as_ubyte(background)
    if background.ndim == 2: background = np.stack([background] * 3, axis = -1)
    if not isinstance(points, list): points = [points]
    for point in points:
        rr, cc = draw.disk(point, radius = 2)
        draw.set_color(background, (rr, cc), color = (255, 0, 0))    
    return img_as_ubyte(background)   

def patch_CLAHE(pred, n_pieces = 8):
    assert pred.ndim == 2
    H, W = pred.shape[0], pred.shape[1]
    h, w = H // n_pieces, W // n_pieces
    for i in range(n_pieces):
        for j in range(n_pieces):
            patch = pred[i*h : (i+1) * h, j*w:(j+1) * w]
            if patch.sum() > 10:
                pred[i*h : (i+1) * h, j*w:(j+1) * w] = equalize_adapthist(patch)
    return pred

def p_near_ps(p1, points, threshold):
    # decide if p is near to any point in points
    for p2 in points:
        if abs(p1[0] - p2[0]) <= threshold and abs(p1[1] - p2[1]) <= threshold: 
            return True
    return False

def remove_nears(points, threshold):
    # remove p in points if they are close
    points = sorted(points, key = lambda x : x[0])
    remove_idx = []
    for i, p in enumerate(points):
        others = points.copy()
        others.pop(i)
        if p_near_ps(p, others, threshold) : remove_idx.append(i)
    points = [points[i] for i in range(len(points)) if i not in remove_idx]    
    return points
    

def get_bbox_cpoints(pred, patch_size = 17):
    # get skel which keeps tiny vessels
    pred = equalize_adapthist(pred)
    thre1 = threshold_otsu(pred)
    pred_bin = pred > thre1
    low_intens = pred * (1-pred_bin)
    low_intens_bin = low_intens > threshold_otsu(low_intens)
    pred_bin_low = low_intens_bin + pred_bin
    skel = thin(pred_bin_low)
    skel = remove_small_objects(skel, 16, connectivity=2)
    skel = skel.astype(bool)
    # get end & branch points
    endpoints = bwmorph(skel, 'endpoints')
    branchpoints = bwmorph(skel, 'branchpoints')
    end_branch = endpoints + branchpoints
    end_branch_points = (np.argwhere(end_branch == True)).tolist()
    # get v-grid points
    all_skel_points = (np.argwhere(skel == True)).tolist()
    height = pred.shape[0]
    grid_width = (patch_size - 1) // 1
    grids = [i*grid_width for i in range((height // grid_width) + 1)]
    v_grid_points = list(filter(lambda x : x[0] in grids, all_skel_points))
    v_grid_points = remove_nears(v_grid_points, threshold = grid_width // 2)
    v_grid_points = list(filter(lambda x : not p_near_ps(x, end_branch_points, threshold = grid_width // 2), v_grid_points))
    # v_grid_point should dist from all end_branch_points for at least grid_width / 2
    # get h-grid points
    h_grid_points = list(filter(lambda x : x[1] in grids, all_skel_points))
    h_grid_points = remove_nears(h_grid_points, threshold = grid_width // 2)
    h_grid_points = list(filter(lambda x : not p_near_ps(x, end_branch_points + v_grid_points, threshold = grid_width // 2), h_grid_points))
    bbox_cpoints = end_branch_points + v_grid_points + h_grid_points
    #bbox_cpoints = sorted(bbox_cpoints, key = lambda x : x[0])
    return skel, bbox_cpoints

def connect_bbox_cpoints(points, narrow = 100, n_links = 3):
        assert narrow < 256, 'param narrow is too large'
        centers = [(i, x[0], x[1]) for i, x in enumerate(points)]

        graph = nx.Graph()
        
        for i, x in enumerate(centers):
            graph.add_node(centers[i][0], center = (centers[i][1], centers[i][2]))
            
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
            
        return nx.DiGraph(graph)    


def make_bbox_graph(pred, n_bbox = None, patch_size = 17, narrow = 100, n_links = 2):
    _, bbox_cpoints = get_bbox_cpoints(pred, patch_size)
    
    if n_bbox is None:
        graph = connect_bbox_cpoints(bbox_cpoints, narrow, n_links)
        return graph
    
    if len(bbox_cpoints) > n_bbox:
        random.shuffle(bbox_cpoints)
        bbox_cpoints = bbox_cpoints[:n_bbox]   
        graph = connect_bbox_cpoints(bbox_cpoints, narrow, n_links)
    
    elif len(bbox_cpoints) < n_bbox:
        graph = connect_bbox_cpoints(bbox_cpoints, narrow, n_links)
        random_cpoints = np.random.randint(low = patch_size // 2, \
                                           high = min(pred.shape[0], pred.shape[1]) - patch_size // 2,\
                                           size = (n_bbox - len(bbox_cpoints), 2)).tolist()
        random_nodes = [(i + len(graph.nodes), p[0], p[1]) for i, p in enumerate(random_cpoints)]    
        
        for node in random_nodes : graph.add_node(node[0], center = (node[1], node[2]))
    return graph

def make_bbox_graph_small(pred, narrow = 10, n_links = 3, scale = 8, patch_size = 17):
    # make bbox graph for downscaled feature map
    # all pixels in downscaled feature map are considered as nodes
    _, cpoints = get_bbox_cpoints(pred, patch_size = patch_size)
    cpoints_d = [(p[0]/scale, p[1]/scale) for p in cpoints]
    cpoints_d_int = [(int(p[0]), int(p[1])) for p in cpoints_d]
    # find valid nodes
    valid_idx = []
    valid_p = []
    for i, p in enumerate(cpoints_d_int):
        if p not in valid_p:
            valid_p.append(p)
            valid_idx.append(i)     
     
    cpoints_d_int_valid = [cpoints_d_int[i] for i in valid_idx]
    graph = connect_bbox_cpoints(cpoints_d_int_valid, narrow, n_links)
    # add other grid nodes back, which are isolate nodes
    new_shape = (pred.shape[0] // scale, pred.shape[1] // scale)
    all_centers = [node[1]['center'] for node in graph.nodes(data = True)]
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            if (i, j) not in all_centers:
                graph.add_node(len(all_centers) + 64*i + j, center = (i, j))
    
    return graph
            

# drives = RetinalDataset('DRIVE').all_data

# for drive in drives:
#     pred = drive.pred
#     bbox = drive.bbox
#     pred = pred[bbox[0]:bbox[2], bbox[1]:bbox[3]]
#     pred = resize(pred, output_shape = [512, 512])
#     start = datetime.now()
#     graph = make_bbox_graph_small(pred)
#     print(graph)
#     end = datetime.now()
#     print(f'time {int((end-start).total_seconds()*1000)} ms')
    #io.imsave(f'test/{drive.ID}_graph.png' ,vis_bbox_graph(graph, pred))
















