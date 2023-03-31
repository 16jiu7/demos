#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:19:57 2023
make graph from pred, but in a faster way:
do not register pieces
do calculate in uint8
eu-dist for neighbor screening
nearest-k eu-dist to get edges
@author: jiu7
"""
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte, img_as_bool
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch


class GraphedImage():
    def __init__(self, pred, mask, N_pieces):
        # VALUES & PRECHECKS
        self.ISP = os.path.join('..', 'images') # debugging image save path
        self.neg_ratio = 0.5 # the least 0.2 pieces will be neglacted
        self.N_pieces = N_pieces
        self.narrow = 100
        self.N_links = 3
        self.intergrate_narrow = 200
        
        self.pred = img_as_ubyte(pred)
        self.mask = img_as_bool(mask)
        self.graph = nx.Graph()
        # REGISTER FUNCTIONS HERE
        self.binarize = binarize
        self.get_slic = get_slic
        self.add_nodes = add_nodes
        self.add_edges = add_edges
        self.visualize_graph = visualize_graph
        self.draw_some_nodes = draw_some_nodes
        self.graph_integrate = graph_integrate
        self.make_edge_id_tensor = make_edge_id_tensor
        # DESCRIBE PIPELINE HERE
        self.certain_pred, self.uncertain_pred, self.certain_threshold = self.binarize(self, True)       
        self.slic_label, self.boundaries = self.get_slic(self, N_pieces = self.N_pieces)
        self.add_nodes(self)
        self.add_edges(self, narrow = self.narrow, N_link = self.N_links, src_list = list(self.graph.nodes), valid_tar_list = list(self.graph.nodes)) 
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        self.graph_integrate(self)
        self.edge_index = self.make_edge_id_tensor(self)
        print('make_graph:', self.graph)
        
    def mark_some_nodes(self, to_draw_list, color):
        self.draw_some_nodes(self, to_draw_list, color = color)
        
def binarize(self, save) -> np.array:
    threshold = threshold_otsu(self.pred)
    certain_pred = self.pred > threshold
    uncertain_pred = np.clip(self.pred - 255*certain_pred, a_min = 0, a_max = None).astype(np.uint8)
    if save:
        io.imsave(os.path.join(self.ISP, 'certain_pred.png'), img_as_ubyte(certain_pred))
        io.imsave(os.path.join(self.ISP, 'uncertain_pred.png'), img_as_ubyte(uncertain_pred))
    return certain_pred, uncertain_pred, threshold

def get_slic(self, N_pieces : int) -> np.array:
    slic_label = slic(np.stack([self.certain_pred, self.certain_pred, self.certain_pred], axis = -1), N_pieces, mask = self.mask)
    certain_pred_boundaries = mark_boundaries(self.certain_pred, slic_label, mode = 'inner',color = (0,0,1)) # blue
    return slic_label, certain_pred_boundaries 

def add_nodes(self) -> None:
    # nodes that has certain_pred
    certain_nodes = list(np.unique(self.slic_label * self.certain_pred))
    certain_nodes.remove(0)
    # nodes that has greater amount of uncertain_pred
    select_mask = self.uncertain_pred > (self.uncertain_pred.max() * self.neg_ratio)
    uncertain_nodes = list(np.unique(self.slic_label * select_mask))
    uncertain_nodes.remove(0)
    # assign node attributes    
    props = regionprops(self.slic_label, self.pred)
    for label in certain_nodes + uncertain_nodes:
        index = label - 1
        center_y, center_x = props[index].centroid
        center = (int(center_y), int(center_x))
        slices = props[index].slice
        node_kind = 'certain' if label in certain_nodes else 'uncertain'
        self.graph.add_node(label, center = center, slices = slices, node_kind = node_kind)
    #print(f'add_nodes: certain & uncertain nodes:[{len(certain_nodes)}, {len(uncertain_nodes)}]')    
    self.certain_nodes, self.uncertain_nodes = certain_nodes, uncertain_nodes

def add_edges(self, narrow, N_link, src_list, valid_tar_list) -> None:
    assert narrow < 256, 'param narrow is too large'
    # only from some scr nodes to some other nodes
    def get_neighbors(src, narrow, valid_tars) -> list:
        select_mask = np.zeros_like(self.certain_pred).astype(bool)
        [rr, cc] = draw.disk(center = self.graph.nodes[src]['center'], radius = narrow, shape = self.pred.shape) 
        select_mask[rr, cc] = True
        neighbors = list(np.unique(select_mask * self.slic_label))
        neighbors = list(set(neighbors) & set(valid_tars))
        try: 
            neighbors.remove(src)
        except: ValueError
        actual_N_link = min(N_link, len(neighbors))
        return neighbors, actual_N_link

    def get_topk_dist_eu(src, tars, k):
        X = np.array([self.graph.nodes[tar]['center'][1] for tar in tars], dtype = np.uint16)
        Y = np.array([self.graph.nodes[tar]['center'][0] for tar in tars], dtype = np.uint16)
        x, y = self.graph.nodes[src]['center'][1], self.graph.nodes[src]['center'][0]
        DST = (X - x) ** 2 + (Y - y) ** 2
        rank = np.argsort(DST)
        tars = np.array(tars, dtype = np.uint16)
        topk_tars = tars[rank[:k]]
        return list(topk_tars)
        
    for label in src_list:
        
        neighbors, actual_N_link = get_neighbors(label, narrow, valid_tars = valid_tar_list)
        targets = get_topk_dist_eu(label, neighbors, actual_N_link)
        edges = [(label, target) for target in targets]
        self.graph.add_edges_from(edges)

def graph_integrate(self):
    # smaller parts are connected to other components
    N_components = nx.number_connected_components(self.graph)
    componets = nx.connected_components(self.graph)
    child_componets = sorted(componets, key = len)[:N_components - 1]
    srcs = []
    for child_componet in child_componets:
        srcs += list(child_componet)
    for src in srcs:
        self.add_edges(self, self.intergrate_narrow, self.N_links, src_list = srcs, valid_tar_list = list(set(self.graph.nodes) - set(srcs)))
    
def make_edge_id_tensor(self):
    # shape = (2, E), data type = torch.long for GAT use
    edges = list(self.graph.edges)
    edges = torch.Tensor(edges).long().transpose(1,0)
    return edges

def draw_some_nodes(self, to_draw_list : list, color, save = True):
    
    def draw_nodes(image : np.array, coords : list, color : str) -> np.array: # image is float btw 0,1
        if color == 'red' : cur_color = (1,0,0)
        elif color == 'blue' : cur_color = (0,0,1) 
        elif color == 'yellow' : cur_color = (1,1,0)
        for coord in coords:
            rr, cc = draw.circle_perimeter(coord[0], coord[1], 2) # radius = 2
            image[rr, cc] = cur_color # color = red
        return image
    
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
                some_centers.append(self.graph.nodes[label]['center'])
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
        pos[i] = (self.graph.nodes[i]['center'][1], self.graph.nodes[i]['center'][0])
    
    # define node colors by their kind
    node_colors = []
    for node in self.graph.nodes:
        if self.graph.nodes[node]['node_kind'] == 'certain':
          node_colors.append('red')
        elif self.graph.nodes[node]['node_kind'] == 'uncertain':
          node_colors.append('green')
    
    nx.draw(self.graph, pos, node_color = node_colors, edge_color='red', width=0.5, node_size=5, alpha=0.5)

    if save_name is not None:
        plt.savefig(os.path.join(self.ISP, save_name), bbox_inches='tight', pad_inches=0, dpi = 600)
    if show_graph:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()           
    
    
# In[]
from data_handeler import RetinalDataset
import datetime
# from os import listdir

# pred_dir = '/home/jiu7/Downloads/LadderNet-master/STARE_results/'
# names = listdir(pred_dir)
# stare = RetinalDataset('STARE').all_data
# for i, data in enumerate(stare):
#       pred = io.imread(pred_dir + data.ID + '.png')[605*2:, :]
#       stare[i].pred = pred
     
# for data in stare:
#     pred = data.pred
#     mask = data.fov_mask
#     starttime = datetime.datetime.now()
#     graphedpred = GraphedImage(pred, mask, 3000)
#     endtime = datetime.datetime.now()
#     print(f'Run time : {endtime - starttime}s')
    
#     graphedpred.visualize_graph(graphedpred, save_name = f'{data.ID}_graph.png') 


# def add_edges(graph, narrow, n_link, intergrate_narrow = 100) -> None:
#     assert narrow < 256, 'param narrow is too large'
    
#     nodes_data = [(node, graph.nodes[node]['center'][0], graph.nodes[node]['center'][1]) for node in graph.nodes]

#     for src in nodes_data:
#         # find neighbors for src

#         neighbors = list(filter(lambda x: (abs(x[1] - src[1]) < narrow) and \
#                            (abs(x[2] - src[2]) < narrow and x[0] != src[0]), nodes_data))
#         actual_n_link = min(len(neighbors), n_link)
#         # get n_link nearest nodes from neighbors
#         X = np.array([neighbor[1] for neighbor in neighbors], dtype = np.uint16)
#         Y = np.array([neighbor[2] for neighbor in neighbors], dtype = np.uint16)
#         x, y = src[1], src[2]
#         DST = (X - x) ** 2 + (Y - y) ** 2
#         rank = np.argsort(DST)
#         neighbor_idxs = [neighbor[0] for neighbor in neighbors]
#         topk_neighbor_idxs = [neighbor_idxs[i] for i in rank[:actual_n_link]]
#         edges = [(src[0], topk_neighbor_idx) for topk_neighbor_idx in topk_neighbor_idxs] 
#         graph.add_edges_from(edges)
#     # connect all components, from smaller parts to the largest component
#         N_components = nx.number_connected_components(graph)
#         componets = nx.connected_components(graph)
#         child_componets = sorted(componets, key = len)[:N_components - 1]
#         #TODO
#     return graph
    
    
# def make_graph(pred, mask, n_pieces, neg_ratio, narrow, n_link):
#     # pred -> certain & uncertain pred
#     threshold = threshold_otsu(pred)
#     certain_pred = pred > threshold
#     uncertain_pred = np.clip(pred - 255*certain_pred, a_min = 0, a_max = None).astype(np.uint8)
#     # run slic on certain_pred
#     slic_label = slic(np.stack([certain_pred]*3, axis = -1), n_pieces, mask = mask)
#     # remove small prediction values, to control edge numbers
#     neg_val = int(uncertain_pred.max() * neg_ratio)
#     uncertain_pred = np.where(uncertain_pred < neg_val, 0, uncertain_pred)
#     # 2 graphs, will merge at the end
#     empties = nx.Graph()
#     fulls = nx.Graph()
#     # add nodes, node attrs: idx, label, bbox, center
#     props = regionprops(slic_label, uncertain_pred + certain_pred)
#     for idx, prop in enumerate(props):
#         if prop.intensity_mean == 0:
#             empties.add_node(idx, label = prop.label, bbox = prop.bbox, center = (int(prop.centroid[0]), int(prop.centroid[1])))
#         else:    
#             fulls.add_node(idx, label = prop.label, bbox = prop.bbox, center = (int(prop.centroid[0]), int(prop.centroid[1])))
#     # add edges, for fulls only
#     fulls = add_edges(fulls, narrow, n_link)
#     fulls.add_nodes_from(empties.nodes(data = True))
    
#     return nx.DiGraph(fulls)
    
# def visualize_graph(background, graph, show_graph=True, save_graph=True, save_name = 'tmp_graph.png') -> None:
#     im = background
#     plt.figure(figsize=(7, 6.05))
#     bg = im.astype(int)*255 if im.dtype == np.float32 else im
    
#     if len(bg.shape)==2:
#         plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
#     elif len(bg.shape)==3:
#         plt.imshow(bg)
#     plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
#     plt.axis((0,700,605,0))
#     pos = {}
    
#     # define node positions
#     for i in graph.nodes:
#         #print(graph.nodes[i])
#         pos[i] = (graph.nodes[i]['center'][1], graph.nodes[i]['center'][0])
    
#     nx.draw_networkx(graph, pos, node_color = 'green', arrowsize = 2, edge_color='red', width=0.5, node_size=5, alpha=0.5, with_labels = False)

#     if save_name is not None:
#         plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi = 400)
#     if show_graph:
#         plt.show()
    
#     plt.cla()
#     plt.clf()
#     plt.close()      
    
# drive = RetinalDataset('STARE', cropped = False).all_data[0]
# starttime = datetime.datetime.now()
# graph = make_graph(drive.pred, drive.fov_mask, 2000, 0.5, 100, 4)    
# endtime = datetime.datetime.now()
# print(f'Run time : {endtime - starttime}s')
# visualize_graph(drive.pred, graph)
    
    
    
    
    
    
    
    
    

