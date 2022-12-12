#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:54:38 2022

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

meta_paths = ["piece_to_piece", "piece_to_pixel", "pixel_to_pixel"]

class GraphedImage():
    def __init__(self, ori, pred, mask, N_pieces):
        # VALUES & PRECHECKS
        self.ori = img_as_float32(ori) 
        self.pred = img_as_float32(pred) 
        assert self.pred is not None, "GraphedImage : pred must exist"
        self.mask = mask
        self.certain_threshold = "auto"
        self.neglact_threshold = 0.01
        self.graph = nx.Graph()
        self.N_pieces = N_pieces
        self.piece_list = self.kind0_list = self.kind1_list = self.kind2_list = []
        self.node_list = [] # CAUTION : node labels start from 1 while piece_list's index starts from 0
        # REGISTER FUNCTIONS HERE
        self.compute_mask = compute_mask
        self.binarize = binarize
        self.get_slic = get_slic
        self.register_pieces = register_pieces
        self.sort_pieces = sort_pieces
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
        
        self.kind0_list, self.kind1_list, self.kind2_list = self.sort_pieces(self)
        #self.draw_some_nodes(self, self.kind2_list, 'red')
        self.kind_all_list = self.kind0_list + self.kind1_list + self.kind2_list
        self.node_list = self.kind1_list + self.kind2_list
        self.add_nodes(self)
        self.add_edges(self, narrow = 200, method = 'shortest', \
                       threshold = 200, N_link = 3) 
        iso_nodes = nx.isolates(self.graph)
        self.graph.remove_nodes_from(list(iso_nodes)) 
        
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
        io.imsave('certain_pred.png', img_as_ubyte(certain_pred))
        io.imsave('uncertain_pred.png', img_as_ubyte(uncertain_pred))
    return certain_pred, uncertain_pred, threshold

def get_slic(self, N_pieces : int, save : bool) -> np.array:
    slic_label = slic(np.stack([self.certain_pred, self.certain_pred, self.certain_pred], axis = -1), N_pieces, mask = self.mask)
    certain_pred_boundaries = mark_boundaries(self.certain_pred, slic_label, mode = 'inner',color = (0,0,1)) # blue
    uncertain_pred_boundaries = mark_boundaries(self.uncertain_pred, slic_label, mode = 'inner',color = (1,0,1))
    if save:
        io.imsave('certain_slic_res.png', img_as_ubyte(certain_pred_boundaries))
        io.imsave('uncertain_slic_res.png', img_as_ubyte(uncertain_pred_boundaries))  
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
        # for kind1 pieces, cores are pixels where uncertain_pred > cut_value
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
        
    assert None not in piece_list
    if not debug:
        return piece_list
    if debug:
        print(f'{len(piece_list)} pieces found, kind0 : {len(kind0_list)}, kind1 : {len(kind1_list)}, kind2 : {len(kind2_list)}')
        return piece_list

def sort_pieces(self) -> list:
    self.kind0_list = [piece.label for piece in self.piece_list if piece.kind == 0] 
    self.kind1_list = [piece.label for piece in self.piece_list if piece.kind == 1] 
    self.kind2_list = [piece.label for piece in self.piece_list if piece.kind == 2] 
    assert len(self.kind0_list) + len(self.kind1_list) + len(self.kind2_list) == len(self.piece_list)       
    return self.kind0_list, self.kind1_list, self.kind2_list

def add_nodes(self) -> None:
    for label in self.node_list:
        index = label - 1
        center_y, center_x = self.piece_list[index].center
        node_kind = self.piece_list[index].kind
        self.graph.add_node(label, y = center_y, x = center_x, node_kind = node_kind)

def get_neighbors(self, node, narrow) -> list:
    '''
    get possible linking nodes (neighbors) for a specific node
    child function for self.add_edges
    '''
    i = node - 1 # trans from node label to according self.piece_list index
    cur_slice = self.piece_list[i].slices
    phi = np.ones_like(self.certain_pred)
    phi[cur_slice] -= 2 * (self.pred > self.neglact_threshold)[cur_slice]

    mask = np.zeros_like(self.pred)
    [rr, cc] = draw.disk(center = self.piece_list[i].center, radius = 50, shape = self.pred.shape) 
    mask[rr, cc] = 1
    speed = self.certain_pred + self.uncertain_pred * mask

    tt = skfmm.travel_time(phi, speed = self.pred, narrow = narrow).astype(np.float32)    
    #io.imsave('tt.png', tt)
    neighbor_mask = np.where(tt.filled(fill_value=0) > 0, 1, 0) 
    #io.imsave('neighbor_mask.png', neighbor_mask)
    neighbors = list(np.unique(neighbor_mask * self.slic_label))
    neighbors.remove(0)
    neighbors = list(set(neighbors) - set(self.kind0_list + [node]))
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

def add_edges(self, narrow : int, method : str, threshold : float, N_link : int) -> None:
    '''
    decide which node(s) in neighbors are to be connected 
    and map that opretion to all kind2 nodes
    '''
    assert method in ['shortest', 'threshold']
    
    if method == 'shortest':

        for label in self.kind2_list + self.kind1_list:
            tt, neighbors = self.get_neighbors(self, label, narrow)
            
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
            #print(f'edge btw {label} and {target} by {target_dist : .3e} {max_dist:.3e}' )
            
    elif method == 'threshold':
        
         for label in self.kind2_list:
             tt, neighbors = self.get_neighbors(self, label, narrow)
             
             tt = (tt - tt.min()) / (tt.max() - tt.min()) # normalization btw 0,1
             tt_filled = tt.filled(fill_value = 1).astype(np.float32)
             
             mean_geo_dists = []
             for neighboor in neighbors:
                 dist = self.get_dist(self, neighboor, tt_filled)
                 mean_geo_dists.append((neighboor , dist)) 
                 
             targets = [mean_geo_dist[0] for mean_geo_dist in mean_geo_dists if mean_geo_dist[-1] < threshold]
             if len(targets) == 0:
                 continue
             else:
                 _ = [self.graph.add_edge(label, target) for target in targets]
                 #print(f'edge btw {label} and {target} by {target_dist : .3e}' )
        
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
    assert len(to_draw_list) > 0, 'to_draw_list is empty'    
    some_centers = []
    for label in to_draw_list:
        some_centers.append(self.piece_list[label - 1].center)
    
    marker_img = draw_nodes(self.boundaries, some_centers, color = color)
    if save:
        io.imsave('marker_img.png', img_as_ubyte(marker_img)) 
        
def visualize_graph(self, show_graph=True, save_graph=True, save_path='graph.png') -> None:
    im = self.boundaries
    graph = self.graph
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
    node_list = list(graph.nodes)
    
    # define node positions
    for i in node_list:
        pos[i] = [graph.nodes[i]['x'], graph.nodes[i]['y']]
    
    # define node colors by their kind
    node_colors = []
    for node in self.graph.nodes:
        if self.graph.nodes[node]['node_kind'] == 0:
          node_colors.append('blue')
        elif self.graph.nodes[node]['node_kind'] == 1:
          node_colors.append('green')
        else:
          node_colors.append('red')
    
    nx.draw(graph, pos, node_color = node_colors, edge_color='red', width=0.5, node_size=5, alpha=0.5)

    if save_graph:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi = 600)
    if show_graph:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()           
    
# if __name__ == '__main__':
    
#     tmp_dir_for_demo = '/home/jiu7/Downloads/LadderNet-master/STARE_results/im0001.png'
#     whole_img = io.imread(tmp_dir_for_demo)
#     gt, pred = whole_img[605:605*2, :], whole_img[605*2:, :]
#     del whole_img

#     graphedpred = GraphedImage(pred)
#     piece_list = graphedpred.piece_list
#     a = graphedpred.graph
#     # print(a)
#     # print(list(a.nodes))
#     graphedpred.draw_graph()
#     # to_draw_list = graphedpred.kind2_list
#     graphedpred.mark_some_nodes(297)
#     # graphedpred.draw_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    