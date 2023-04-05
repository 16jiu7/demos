#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:27:15 2023

@author: jiu7
"""
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, label
from skimage.util import img_as_ubyte, img_as_bool
from skimage.filters import threshold_minimum, threshold_otsu
from skimage import draw
from skimage.morphology import remove_small_objects
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from data_handeler import RetinalDataset
import datetime
from operator import attrgetter

hrf_0 = RetinalDataset('STARE').all_data[5]
pred = hrf_0.pred
mask = hrf_0.fov_mask.astype(bool)
pred = pred * mask
io.imsave('pred_test.png', pred)
print(pred.max())
certain_pred = pred >= 128
print((255*certain_pred).max())
uncertain_pred = np.clip(pred - 255 * certain_pred, 0, 255)
io.imsave('certain_pred.png', img_as_ubyte(certain_pred))
io.imsave('uncertain_pred.png', img_as_ubyte(uncertain_pred))

certain_pred = remove_small_objects(certain_pred, min_size = 16)
iso_labels = label(certain_pred)

iso_regions = regionprops(iso_labels)

small_iso_regions = list(filter(lambda x : max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17 ,iso_regions))
large_iso_regions = filter(lambda x : min(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) > 17  ,iso_regions)
print('small isolate: ',len(small_iso_regions))
certain_pred_only_large = np.zeros(shape = certain_pred.shape, dtype = bool)
for region in large_iso_regions: certain_pred_only_large[region.slice] += region.image
io.imsave('certain_pred_only_large.png', img_as_ubyte(certain_pred_only_large))

slic_label = slic(certain_pred_only_large, 500, mask = certain_pred_only_large, enforce_connectivity = False, compactness = 300)
boundaries = mark_boundaries(certain_pred, slic_label, color = (1,0,1))
io.imsave('slic_only_in_certain_pred.png', img_as_ubyte(boundaries))   
slic_regions = regionprops(slic_label)
small_slic_regions = list(filter(lambda x : max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17 ,slic_regions))
large_slic_regions = list(filter(lambda x : min(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) > 17  ,slic_regions))
print('small slic: ',len(small_slic_regions))
print('large slic: ',len(large_slic_regions))
for region in large_slic_regions:
    h, w = region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]
    print(region.label, h, w)
    io.imsave(f'{region.label}.png', img_as_ubyte(region.image))
# In[]

def get_nodes_from_pred(pred, params = {'threshold': 'min', 'neg_iso_size': 6, 'n_slic_pieces': 500}, verbose = True):
    assert pred.dtype == np.uint8
    if params['threshold'] == 'min':
        threshold = threshold_minimum(pred)
    certain_pred = pred >= threshold
    certain_pred = remove_small_objects(certain_pred, min_size = params['neg_iso_size'])
    
    all_bboxes = []
    iso_label = label(certain_pred)
    iso_regions = regionprops(iso_label)
    small_iso_bboxes = [x.bbox for x in iso_regions if max(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) <= 17]
    all_bboxes += small_iso_bboxes
    
    large_isos = certain_pred.copy()
    for x in small_iso_bboxes: large_isos[x[0]:x[2], x[1]:x[3]] = 0
    slic_label = slic(large_isos, 500, mask = large_isos, enforce_connectivity = False, compactness = 100)
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
    if verbose : print('number of bboxes: ', len(small_bboxes))
    return small_bboxes, certain_pred

def bbox_vis(bboxes, background, save_dir):
    for bbox in bboxes:
        start = (bbox[0], bbox[1])
        end = (bbox[2], bbox[3])
        rr,cc = draw.rectangle_perimeter(start, end)
        draw.set_color(background, (rr, cc), color = (255,0,0))
    io.imsave(save_dir, img_as_ubyte(background))    

hrf = RetinalDataset('STARE').all_data
for data in hrf:
    pred = data.pred
    mask = data.fov_mask.astype(bool)
    pred = pred * mask
    bboxes, certain_pred = get_nodes_from_pred(pred)
    bbox_vis(bboxes, np.stack([certain_pred]*3, axis = -1), save_dir = f'{data.ID}_bbox_vis.png')
    bbox_sizes = [(x[2] - x[0], x[3] - x[1]) for x in bboxes]



















