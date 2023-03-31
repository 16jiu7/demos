#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:32:18 2023

@author: jiu7
"""
def MakeNodesGT(graph, single_data, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # make GT for node vessel density prediction
    nodes_gt = []
    gt = np.float32(single_data.gt / 255.)
    for node in graph.nodes: 
        slices = graph.nodes[node]['slices']
        mass = np.sum(gt[slices])
        area = (slices[0].stop - slices[0].start) * (slices[1].stop - slices[1].start)
        nodes_gt.append(mass / area) 
    nodes_gt = torch.Tensor(nodes_gt).float()    
    return nodes_gt.to(device)    
      
def GetNodeFeats(cnn_feats, roi_head, slic_label, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    r_props = regionprops(slic_label)
    boxes = torch.zeros(size = [len(r_props), 5], dtype = torch.float)
    
    for idx, prop in enumerate(r_props):
        box = torch.Tensor([0] + list(prop.bbox)).float()
        boxes[idx] = box    
    cnn_feats = cnn_feats.to(device)
    roi_head.to(device)
    boxes = boxes.to(device)
    node_feats = roi_head(cnn_feats, boxes)
    return node_feats

def RunForwardPass(cnn, gnn, single_data, slic_label, graph, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # run CNN encoder
    concat_cnn_feats, side_outs = GetConcatCNNFeats(cnn, single_data)
    node_feats = GetConcatNodeFeats(concat_cnn_feats, graph, slic_label)
    edges = GetRelabeledGraphEdges(graph)
    node_feats, edges = node_feats.to(device), edges.to(device)
    # do a GNN forward pass, get node feats and node prediction results
    gnn.to(device)
    (node_feats_gat, _), (node_density_logits, _) = gnn((node_feats, edges))
    node_feats_gat = node_feats_gat.reshape(-1, node_feats_gat.shape[-1] // GNN_N_HEADS, GNN_N_HEADS)
    node_feats_gat = node_feats_gat.mean(-1) # do mean average on heads dim
    
    shallows, bottom = side_outs[:4], side_outs[-1]
    bottom_replaced = FuseNodeCenterFeats(graph, bottom, node_feats_gat, single_data.gt.shape)
    final_logits = cnn.run_decoder(bottom_replaced, shallows)
    
    return final_logits, node_density_logits # final logits and gnn predictions for node vessel density
  