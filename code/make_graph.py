#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:54:38 2022
this class perform slic on predicted vessel map and connect slic pieces as nodes in an nx graph

node sampling scheme:
    1.pred -> uncertain + certain, using otsu
    2.apply SLIC on certain
    3.sample kind2 nodes, sum(certain[node_pixels]) > 0
    4.sample kind1 nodes, mean(uncertain[node_pixels]) + greatest k% scheme
    5.other slic pieces are not considered as nodes
    
edge linking scheme:
    1.kind2 nodes to other nodes, neighbor_screening + geo_dist + nearest k scheme
    2.(for isolate nodes after 1.) kind1 nodes to other nodes, eu_dist + nearest k scheme



@author: jiu7
"""
import os, sys
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, find_contours
from skimage.util import img_as_float32, img_as_ubyte
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
from skimage import color
from skimage import morphology
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skfmm

class GraphedImage():
    def __init__(self, ori, pred, mask, N_pieces):
        # VALUES & PRECHECKS
        self.ISP = os.path.join('..', 'images') # debugging image save path
        self.ori = img_as_float32(ori) 
        self.pred = img_as_float32(pred) 
        assert self.pred is not None, "GraphedImage : pred must exist"
        self.mask = mask
        self.certain_threshold = "auto"
        self.neglact_threshold = 0.01
        self.kind1_ratio = 0.5 
        # 0.5 of kind2 pieces with greater intensity on uncertain_pred are selected as kind2 nodes
        self.graph = nx.Graph()
        self.N_pieces = N_pieces
        self.piece_list = self.kind0_list = self.kind1_list = self.kind2_list = []
        self.kind1_node_list = []
        # REGISTER FUNCTIONS HERE
        self.compute_mask = compute_mask
        self.binarize = binarize
        self.get_slic = get_slic
        self.register_pieces = register_pieces
        self.add_nodes = add_nodes
        self.add_edges = add_edges
        self.visualize_graph = visualize_graph
        self.draw_some_nodes = draw_some_nodes
        self.get_neighbors = get_neighbors
        self.get_dist = get_dist
        #self.get_dist_shortest = get_dist_shortest
        # DESCRIBE PIPELINE HERE
        self.mask = self.compute_mask(self.ori) if self.mask is None else self.mask
        if self.certain_threshold == "auto":
            self.certain_pred, self.uncertain_pred, self.certain_threshold = \
            self.binarize(self, method = 'otsu', hard_threshold = 0.5, save = True)       
        self.slic_label, self.boundaries = \
        self.get_slic(self, N_pieces = self.N_pieces ,save = True)
        self.piece_list = self.register_pieces(self, debug = True)
        self.actual_N_pieces = len(self.piece_list)
        
        #self.draw_some_nodes(self, self.kind2_list, 'red')
        self.kind_all_list = [i for i in range(1, self.actual_N_pieces + 1)]
        self.add_nodes(self)
        self.mark_some_nodes([self.kind2_list, self.kind1_node_list], color = ['red', 'yellow'])
        self.add_edges(self, narrow = 200, N_link = 3) 
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        
    def draw_graph(self):
        self.visualize_graph(self)
    def mark_some_nodes(self, to_draw_list, color = 'red'):
        self.draw_some_nodes(self, to_draw_list, color = color)

def compute_mask(ori):
    ori = img_as_float32(ori)
    mask = morphology.remove_small_holes(morphology.remove_small_objects(color.rgb2gray(ori) < 0.7, 500),500) \
    if ori.ndim == 3 else\
    morphology.remove_small_holes(morphology.remove_small_objects(ori < 0.7, 500),500)
    return mask
        
def binarize(self, method : str, hard_threshold : float, save : bool) -> np.array:
    assert method in ['hard threshold', 'otsu'], 'invalid binarize method'
    pred = img_as_float32(self.pred)
    if method == 'otsu':
        threshold = threshold_otsu(pred)
    certain_pred = img_as_float32(pred > threshold)
    uncertain_pred = img_as_float32(np.clip(pred - certain_pred, a_min = 0, a_max = None))
    if save:
        io.imsave(os.path.join(self.ISP, 'certain_pred.png'), img_as_ubyte(certain_pred))
        io.imsave(os.path.join(self.ISP, 'uncertain_pred.png'), img_as_ubyte(uncertain_pred))
    return certain_pred, uncertain_pred, threshold

def get_slic(self, N_pieces : int, save : bool) -> np.array:
    slic_label = slic(np.stack([self.certain_pred, self.certain_pred, self.certain_pred], axis = -1), N_pieces, mask = self.mask)
    certain_pred_boundaries = mark_boundaries(self.certain_pred, slic_label, mode = 'inner',color = (0,0,1)) # blue
    uncertain_pred_boundaries = mark_boundaries(self.uncertain_pred, slic_label, mode = 'inner',color = (1,0,1))
    if save:
        io.imsave(os.path.join(self.ISP, 'certain_slic_res.png'), img_as_ubyte(certain_pred_boundaries))
        io.imsave(os.path.join(self.ISP, 'uncertain_slic_res.png'), img_as_ubyte(uncertain_pred_boundaries))
    return slic_label, certain_pred_boundaries 

class Piece():
    def __init__(self, label : int, kind : int, center : list, slices : slice, coords : list, certains : list, uncertains : list, neglactbales :list):
        self.label = label
        self.kind = kind
        self.center = center
        self.slices = slices
        self.coords = coords
        self.certains = certains
        self.uncertains = uncertains
        self.neglactbales = neglactbales
        # for kind2 pieces, cores are pixels where certain_pred == 1
        # for kind1 pieces, cores are pixels where uncertain_pred > self.neglact_threshold
        # kind0 pieces have no core
         
def register_pieces(self, debug : bool) -> list:
    
    certain_mask = self.certain_pred.astype(bool)
    kind2_list = np.unique(self.slic_label * certain_mask)
    kind2_list = kind2_list.tolist()
    kind2_list.remove(0)

    uncertain_mask = np.where(self.uncertain_pred > self.neglact_threshold, True, False)
    kind1_list = np.unique(self.slic_label * uncertain_mask)
    kind1_list = list(set(kind1_list) - set(kind2_list))
    kind1_list.remove(0)

    slic_props = regionprops(self.slic_label, self.certain_pred)
    kind0_list = [i for i in range(1, len(slic_props) + 1) if i not in kind1_list + kind2_list]
    assert set(kind2_list + kind1_list + kind0_list) == set([i for i in range(1, len(slic_props) + 1)])
    piece_list = [None] * len(slic_props)
    
    for label in kind2_list:
        index = label - 1
        piece = slic_props[index] # note that slic label starts from 1 while slic_props's index from 0
        y, x = piece.centroid
        certain_pixels = uncertain_pixels = neglactbales = []
        for coord in piece.coords:
            intensity = self.pred[coord[0], coord[1]]
            if intensity > self.certain_threshold:
                certain_pixels.append(list(coord))   
            elif intensity >= self.neglact_threshold:
                uncertain_pixels.append(list(coord))
            else:
                neglactbales.append(list(coord))
        tmp = Piece(label = label, kind = 2, center = [int(y), int(x)], 
        slices = piece.slice, coords = list(piece.coords), certains = certain_pixels,
        uncertains = uncertain_pixels, neglactbales = neglactbales)  
        piece_list[index] = tmp
    
    for label in kind1_list:
        index = label - 1
        piece = slic_props[index]
        y, x = piece.centroid
        certain_pixels = uncertain_pixels = neglactbales = []
        for coord in piece.coords:
            intensity = self.pred[coord[0], coord[1]]
            if intensity >= self.neglact_threshold:
                uncertain_pixels.append(list(coord))
            else:
                neglactbales.append(list(coord))
        tmp = Piece(label = label, kind = 1, center = [int(y), int(x)], 
        slices = piece.slice, coords = list(piece.coords), certains = certain_pixels,
        uncertains = uncertain_pixels, neglactbales = neglactbales)        
        piece_list[index] = tmp
        
    for label in kind0_list:
        index = label - 1
        piece = slic_props[index]
        y, x = piece.centroid
        certain_pixels = uncertain_pixels = neglactbales = []
        neglactbales = piece.coords
        tmp = Piece(label = label, kind = 0, center = [int(y), int(x)], 
        slices = piece.slice, coords = list(piece.coords), certains = certain_pixels,
        uncertains = uncertain_pixels, neglactbales = neglactbales)        
        piece_list[index] = tmp   
        
    # hang up results
    self.kind0_list, self.kind1_list, self.kind2_list = kind0_list, kind1_list, kind2_list    
        
    assert None not in piece_list
    if not debug:
        return piece_list
    if debug:
        print(f'register_pieces: {len(piece_list)} pieces, kind012: [{len(kind0_list)}, {len(kind1_list)}, {len(kind2_list)}]')
        return piece_list

def add_nodes(self) -> None:
    # delete kind2 nodes that don't have enough uncertain value from node_list
    props = regionprops(self.slic_label, intensity_image = self.uncertain_pred)
    intensities = []
    for label in self.kind1_list:
        index = label - 1
        intensity = props[index].intensity_mean
        intensities.append((label, intensity))
    intensities = sorted(intensities, key = lambda x : x[-1])
    N_select = int(len(self.kind1_list) * self.kind1_ratio)
    intensities = intensities[N_select:]
    self.kind1_node_list = [x[0] for x in intensities]
    node_list = self.kind2_list + self.kind1_list # not the node_list in the resultant graph
    print(f'add_nodes: kind12 nodes:[{len(self.kind1_node_list)}, {len(self.kind2_list)}]')
        
    for label in node_list:
        index = label - 1
        center_y, center_x = self.piece_list[index].center
        slices = self.piece_list[index].slices
        node_kind = self.piece_list[index].kind
        self.graph.add_node(label, y = center_y, x = center_x, slices = slices, node_kind = node_kind)  

def get_neighbors(self, node : int, narrow : int, valid_list : list) -> list:
    '''
    get possible linking nodes (neighbors) for a specific node
    child function for self.add_edges
    valid_list : only nodes in this list can be neighbors
    '''
    i = node - 1 # trans from node label to according self.piece_list index
    cur_slice = self.piece_list[i].slices
    phi = np.ones_like(self.certain_pred)
    phi[cur_slice] -= 2 * (self.pred > self.neglact_threshold)[cur_slice]

    # mask = np.zeros_like(self.pred)
    # [rr, cc] = draw.disk(center = self.piece_list[i].center, radius = 50, shape = self.pred.shape) 
    # mask[rr, cc] = 1
    # speed = self.certain_pred + self.uncertain_pred * mask

    tt = skfmm.travel_time(phi, speed = self.pred, narrow = narrow).astype(np.float32)    
    #io.imsave('tt.png', tt)
    neighbor_mask = np.where(tt.filled(fill_value=0) > 0, 1, 0) 
    #io.imsave('neighbor_mask.png', neighbor_mask)
    neighbors = list(np.unique(neighbor_mask * self.slic_label))
    neighbors.remove(0)
    neighbors = list(set(neighbors) & set(valid_list))
    neighbors.remove(node)
    assert len(neighbors) > 0, f'no neighboor were found for node {node}'
    assert node not in neighbors
    #print(f'{len(neighbors)} neighboors are selected for node {node}')
    #print(sorted(neighbors))
    return tt, neighbors

def get_dist(self, node : int, tt : np.array) -> float:
    '''
    get mean_geo_dist for a specific slic piece in a specific tt map
    child function for self.add_egdes
    '''
    index = node - 1
    sum_dist = sum_intensity = 0
    coords = self.piece_list[index].certains + self.piece_list[index].uncertains
    for coord in coords:
        sum_dist += tt[coord[0], coord[1]]
    return (sum_dist / len(coords)).astype(np.float32)

# def get_dist_shortest(self, node : int, tt : np.array) -> float:
#     '''
#     get the shortest geo-dist for a specific slic piece in a specific tt map
#     child function for self.add_egdes
#     '''
#     index = node - 1
#     coords = self.piece_list[index].certains
#     assert len(coords) > 0
#     dists = []
#     for coord in coords:
#         dists.append(tt[coord[0], coord[1]])
#     min_dist = min(dists)    
#     return min_dist.astype(np.float32)

def add_edges(self, narrow : int, N_link : int) -> None:

    for label in self.kind2_list:
        tt, neighbors = self.get_neighbors(self, label, narrow, valid_list = self.kind1_list + self.kind2_list)
        
        tt = (tt - tt.min()) / (tt.max() - tt.min()) # normalization btw 0,1
        tt_filled = tt.filled(fill_value = 1).astype(np.float32)
        
        n_link = min(len(neighbors), N_link)
        mean_geo_dists = []
        
        for neighboor in neighbors:
            dist = self.get_dist(self, neighboor, tt_filled)
            mean_geo_dists.append((neighboor, dist)) 
            
        mean_geo_dists = sorted(mean_geo_dists, key = lambda x : x[-1])    
        targets = mean_geo_dists[:n_link]
        edges = [(label, target[0]) for target in targets]
        self.graph.add_edges_from(edges)
        N_stage1_edges = len(self.graph.edges)
    print(f'add edges: kind2 -> *, {N_stage1_edges} edges added')
        
    # further add egdes for nodes that are still isolate (they should only be kind1 nodes)
    # and only elements in kind1_node_list are considered as starting nodes
    isolates = list(nx.isolates(self.graph))
    srcs = list(set(self.kind1_node_list) & set(isolates))
    targets = self.kind1_node_list + self.kind2_list
    def get_dist_eu(node1, node2):
        diffx = self.graph.nodes[node1]['x'] - self.graph.nodes[node2]['x']
        diffy = self.graph.nodes[node1]['y'] - self.graph.nodes[node2]['y']
        return int(diffx**2 + diffy**2)
    for label in srcs:
        tmp_targets = targets.copy()
        tmp_targets.remove(label)
        dists = [(target, get_dist_eu(label, target)) for target in tmp_targets]
        dists = sorted(dists, key = lambda x : x[-1])
        edges = [(label, dist[0]) for dist in dists[:N_link]]
        self.graph.add_edges_from(edges)
    N_stage2_edges = len(self.graph.edges) - N_stage1_edges    
    print(f'add edges: kind1 -> *, {N_stage2_edges} edges added')
        
def draw_nodes(image : np.array, coords : list, color : str) -> np.array: # image is float btw 0,1
    if color == 'red' : cur_color = (1,0,0)
    elif color == 'blue' : cur_color = (0,0,1) 
    elif color == 'yellow' : cur_color = (1,1,0)
    for coord in coords:
        rr, cc = draw.circle_perimeter(coord[0], coord[1], 2) # radius = 2
        image[rr, cc] = cur_color # color = red
    return image

def draw_some_nodes(self, to_draw_list : list, color, save = True):
    if isinstance(to_draw_list, int):
        to_draw_list = [to_draw_list]
    if isinstance(color, list):
        assert len(to_draw_list) == len(color), 'to_draw_list and color should have equal length'
    assert len(to_draw_list) > 0, 'to_draw_list is empty' 
    
    if not isinstance(color, list):
        some_centers = []
        for label in to_draw_list:
            some_centers.append(self.piece_list[label - 1].center)
        marker_img = draw_nodes(self.boundaries, some_centers, color = color)    
    else:
        for i, sub_color in enumerate(color):
            tmp_img = self.boundaries
            some_centers = []
            for label in to_draw_list[i]:
                some_centers.append(self.piece_list[label - 1].center)
            tmp_img = draw_nodes(tmp_img, some_centers, color = sub_color)    
        marker_img = tmp_img
    if save:
        io.imsave(os.path.join(self.ISP, 'marker_img.png'), img_as_ubyte(marker_img))
        
def visualize_graph(self, show_graph=True, save_graph=True, save_name = 'graph.png') -> None:
    im = self.boundaries
    plt.figure(figsize=(7, 6.05))
    
    if im.dtype == np.float32:
        bg = im.astype(int)*255
    else:
        bg = im
    
    if len(bg.shape)==2:
        plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    elif len(bg.shape)==3:
        plt.imshow(bg)
    plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    plt.axis((0,700,605,0))
    pos = {}
    
    # define node positions
    for i in self.graph.nodes:
        pos[i] = [self.graph.nodes[i]['x'], self.graph.nodes[i]['y']]
    
    # define node colors by their kind
    node_colors = []
    for node in self.graph.nodes:
        if self.graph.nodes[node]['node_kind'] == 1:
          node_colors.append('green')
        elif self.graph.nodes[node]['node_kind'] == 2:
          node_colors.append('red')
    
    nx.draw(self.graph, pos, node_color = node_colors, edge_color='red', width=0.5, node_size=5, alpha=0.5)

    if save_name is not None:
        plt.savefig(os.path.join(self.ISP, save_name), bbox_inches='tight', pad_inches=0, dpi = 600)
    if show_graph:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
