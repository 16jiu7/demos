#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:52:17 2023

color_spaces = [rgb, hsv, lab]
calcus = [means, maxs, stds, contrasts, skews, entropies]
3 * 5 * 3 = 45 feats

edge_images = [G_ce, scharr, laplace](scharr and laplace are from G_ce)
calcus = [means, maxs, stds]
glcm_angles = [0, 45, 90, 135]
glcm_feats = [‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’]
glszm_feats: 16, the glszm matrix is rotational independent, so only 1 matrix for a patch
glrlm_angles = [0, 45, 90, 135]
glrlm_feats: 16
3 * 3 * 1 + 4 * 6 + 16 = 49 feats

@author: jiu7
"""
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import scharr, laplace, difference_of_gaussians
from skimage.exposure import equalize_adapthist # CLAHE 
from scipy.stats import shapiro, normaltest
from pyfeats import fos, glcm_features, glrlm_features
from Grinsven_color_enhence import GrinsvenColorEnhance
from mahotas.features import haralick


def hand_feats_extractor(img: np.uint, label_slices: list) -> dict:
    '''
    img: the whole image
    label_slices: [(node_label, slices),... ]
    output: [(node_label, feats),...]
    '''
    output = []
    # transform by whole image then extract patches
    img_hsv = img_as_ubyte(rgb2hsv(img))
    img_lab = rgb2lab(img) 
    img_lab = np.stack([img_lab[:,:,0] * 2.55, img_lab[:,:,1] + 128, img_lab[:,:,2] + 128], axis = -1).astype(np.uint)
    G_ce = equalize_adapthist(GrinsvenColorEnhance(img, pic_width = img.shape[1])[:,:,1])
    G_ce = img_as_ubyte(G_ce)
    # extract patches then transform
    for label, slices in label_slices:
        patch_rgb = img[slices]
        patch_hsv = img_hsv[slices]
        patch_lab = img_lab[slices]
        
        r_feats, _ = fos(patch_rgb[:,:,0], None)
        g_feats, _ = fos(patch_rgb[:,:,1], None)
        b_feats, _ = fos(patch_rgb[:,:,2], None)
        h_feats, _ = fos(patch_hsv[:,:,0], None)
        s_feats, _ = fos(patch_hsv[:,:,1], None)
        v_feats, _ = fos(patch_hsv[:,:,2], None)
        l_feats, _ = fos(patch_lab[:,:,0], None)
        a_feats, _ = fos(patch_lab[:,:,1], None)
        bb_feats, _ = fos(patch_lab[:,:,2], None)
        fos_feats = np.stack([r_feats, g_feats, b_feats, h_feats, s_feats, v_feats, l_feats, a_feats, bb_feats], axis = 0)
        fos_feats = fos_feats.reshape(144,)
        
        patch_G_ce = G_ce[slices]
        glcm_feats,_ ,_ ,_ = glcm_features(patch_G_ce, None) # only mean features of 4 directions
        glrlm_feats, _ = glrlm_features(patch_G_ce, None)
        #feats = np.concatenate((fos_feats, glcm_feats, glrlm_feats), axis = 0)
        feats = fos_feats
        feats = np.reshape(feats, (1, feats.size))
        output.append(feats)
    feats = np.stack(output, axis = 1)[0,...]
    return feats
    
















