#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:54:38 2022

@author: jiu7
"""
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
from networkx.drawing.layout import rescale_layout
import skfmm
from easydict import EasyDict as edict

class GraphedImage():
    def __init__(self, pred):
        # VALUES
        self.pred = pred
        self.graph = nx.Graph()
        self.N_pieces = None
        self.piece_list = self.kind0_list = self.kind1_list = self.kind2_list = []
        self.node_list = [] # CAUTION : node labels start from 1
        # FUNCTIONS
        self.binarize = binarize
        self.get_slic = get_slic
        self.register_pieces = register_pieces
        self.sort_pieces = sort_pieces
        self.add_nodes = add_nodes
        self.add_edges = add_edges
        self.visualize_graph = visualize_graph
        self.draw_some_nodes = draw_some_nodes
        self.get_neighbors = _get_neighbors
        # PIPELINE
        self.certain_pred, self.uncertain_pred = \
            self.binarize(self, method = 'otsu', hard_threshold = 0.5, save = True)         
        self.slic_label, self.boundaries = \
            self.get_slic(self, N_pieces = 2000 ,save = True)
        self.piece_list = self.register_pieces(self, cut_value = 1e-2, debug = True)
        self.N_pieces = len(self.piece_list)
        
        self.kind0_list, self.kind1_list, self.kind2_list = self.sort_pieces(self)
        self.node_list = self.kind2_list
        self.add_nodes(self)
        neighbors = self.get_neighbors(self, node = 88, narrow = 50)
        self.mark_some_nodes(to_draw_list = 88, color = 'yellow')
        self.mark_some_nodes(neighbors, color = 'red')
            
    def draw_graph(self):
        self.visualize_graph(self)
    def mark_some_nodes(self, to_draw_list, color = 'red'):
        self.draw_some_nodes(self, to_draw_list, color = color)
        
def binarize(self, method : str, hard_threshold : float, save : bool) -> np.array:
    assert method in ['hard threshold', 'otsu'], 'invalid binarize method'
    assert self.pred.dtype == np.uint8
    pred = img_as_float32(self.pred)
    if method == 'otsu':
        threshold = threshold_otsu(pred)
    certain_pred = img_as_float32(pred > threshold)
    uncertain_pred = img_as_float32(np.clip(pred - certain_pred, a_min = 0, a_max = None))
    if save:
        io.imsave('certain_pred.png', img_as_ubyte(certain_pred))
        io.imsave('uncertain_pred.png', img_as_ubyte(uncertain_pred))
    return certain_pred, uncertain_pred

def get_slic(self, N_pieces : int, save : bool) -> np.array:
    slic_label = slic(np.stack([self.certain_pred, self.certain_pred, self.certain_pred], axis = -1), N_pieces)
    certain_pred_boundaries = mark_boundaries(self.certain_pred, slic_label, mode = 'inner',color = (0,0,1)) # blue
    uncertain_pred_boundaries = mark_boundaries(self.uncertain_pred, slic_label, mode = 'inner',color = (1,0,1))
    if save:
        io.imsave('certain_slic_res.png', img_as_ubyte(certain_pred_boundaries))
        io.imsave('uncertain_slic_res.png', img_as_ubyte(uncertain_pred_boundaries))  
    return slic_label, certain_pred_boundaries 

class Piece():
    def __init__(self, label : int, kind : int, center : list, slices : slice, coords : list, core : list):
        self.label = label
        self.kind = kind # 0:almost no positive, 1:only uncertain positive, 2:certain positive
        self.center = center
        self.slices = slices
        self.coords = coords
        self.core = core
        
def register_pieces(self, cut_value : float, debug : bool) -> list:
    piece_list = []
    slic_props = regionprops(self.slic_label, self.certain_pred)
    uncertain_props = regionprops(self.slic_label, self.uncertain_pred)
    
    for piece_id, piece in enumerate(slic_props):
        # centers
        y, x = piece.centroid
        # coords and core of kind 2 piece    
        core_pixels = []
        for coord in piece.coords:
            if self.certain_pred[coord[0], coord[1]] == 0:
                continue
            else:
                core_pixels.append(list(coord))        
        # kind           
        if len(core_pixels) > 0:
            kind = 2
            
        elif len(core_pixels) == 0:
            cur_intensity_max = uncertain_props[piece_id].intensity_max
            if cur_intensity_max > cut_value:
                kind = 1
            else : kind = 0
        # core of kind 1 piece
        if kind == 1:
            core_pixels = []
            for coord in piece.coords:
                if self.certain_pred[coord[0], coord[1]] < cut_value:
                    continue
                else:
                    core_pixels.append(list(coord))
        
        tmp = Piece(label = piece.label, kind = kind, center = [int(y), int(x)], slices = piece.slice, coords = list(piece.coords), core = core_pixels)        
        piece_list.append(tmp)
    
    assert len(slic_props) == len(piece_list)
    
    if not debug:
        return piece_list
    if debug:
        N_kind0 = N_kind1 = N_kind2 = 0
        for piece in piece_list:
            if piece.kind == 0:
                N_kind0 += 1
            if piece.kind == 1:
                N_kind1 += 1
            elif piece.kind == 2:
                N_kind2 += 1
        assert N_kind0 + N_kind1 + N_kind2 == len(piece_list)        
        print(f'{len(piece_list)} pieces found, kind0 : {N_kind0}, kind1 : {N_kind1}, kind2 : {N_kind2}')
        return piece_list

def sort_pieces(self) -> list:
    self.kind0_list = [piece.label for piece in self.piece_list if piece.kind == 0] 
    self.kind1_list = [piece.label for piece in self.piece_list if piece.kind == 1] 
    self.kind2_list = [piece.label for piece in self.piece_list if piece.kind == 2] 
    assert len(self.kind0_list) + len(self.kind1_list) + len(self.kind2_list) == len(self.piece_list)       
    return self.kind0_list, self.kind1_list, self.kind2_list

def add_nodes(self) -> None:
    for label in self.node_list:
        center_y, center_x = self.piece_list[label - 1].center
        self.graph.add_node(label, y = center_y, x = center_x)

def _get_neighbors(self, node, narrow) -> list:
    i = node - 1 # trans from node label to according self.piece_list index
    cur_slice = self.piece_list[i].slices
    phi = np.ones_like(self.certain_pred)
    phi[cur_slice] -= 2 * self.certain_pred[cur_slice]

    tt = skfmm.travel_time(phi, speed = self.certain_pred, narrow = narrow).astype(np.float32)    
    neighbor_mask = np.where(tt.filled(fill_value=0) > 0, 1, 0) 
    io.imsave('neighbor_mask.png', neighbor_mask)
    neighbors = list(np.unique(neighbor_mask * self.slic_label))
    print(neighbors)
    neighbors.remove(node)
    neighbors.remove(0)
    
    assert len(neighbors) > 0, f'no neighboor were found for node {node}'
    print(f'{len(neighbors)} neighboors are selected for node {node}')
    print(sorted(neighbors))
    return neighbors

def add_edges(self) -> None:
    pass

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
        
def visualize_graph(self, show_graph=True, save_graph=True, \
                    num_nodes_each_type=None, custom_node_color=None, \
                    tp_edges=None, fn_edges=None, fp_edges=None, \
                    save_path='graph.png'):
    im = self.certain_pred
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
    for i in node_list:
        pos[i] = [graph.nodes[i]['x'], graph.nodes[i]['y']]
    
    if custom_node_color is not None:
        node_color = custom_node_color
    else:
        if num_nodes_each_type is None:
            node_color = 'b'
        else:
            if not (graph.number_of_nodes()==np.sum(num_nodes_each_type)):
                raise ValueError('Wrong number of nodes')
            #node_color = [VIS_NODE_COLOR[0]]*num_nodes_each_type[0] + [VIS_NODE_COLOR[1]]*num_nodes_each_type[1] 

    nx.draw(graph, pos, node_color='green', edge_color='white', width=1, node_size=10, alpha=0.5)

    if save_graph:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi = 600)
    if show_graph:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()           
    
if __name__ == '__main__':
    
    tmp_dir_for_demo = '/home/jiu7/Downloads/LadderNet-master/STARE_results/im0001.png'
    whole_img = io.imread(tmp_dir_for_demo)
    gt, pred = whole_img[605:605*2, :], whole_img[605*2:, :]
    del whole_img

    graphedpred = GraphedImage(pred)
    piece_list = graphedpred.piece_list
    # to_draw_list = graphedpred.kind2_list
    # graphedpred.mark_some_nodes(to_draw_list)
    # graphedpred.draw_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    