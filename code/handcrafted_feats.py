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
import skimage.io as io
from skimage.util import img_as_ubyte
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import scharr, laplace, difference_of_gaussians
from skimage.exposure import equalize_adapthist # CLAHE 
from skimage.measure import regionprops, regionprops_table
from scipy.stats import shapiro, normaltest
from pyfeats import fos, glcm_features, glrlm_features, glszm_features
from Grinsven_color_enhence import GrinsvenColorEnhance
from mahotas.features import haralick

def compute_all_geo_feats(label_img, intensity_img, needed_labels):
    if not isinstance(needed_labels, list):
        needed_labels = [needed_labels]
    props = regionprops(label_img, intensity_img)
    for label_i in needed_labels:
        pass
    
    return 
    
    

def hand_feats_extractor(img: np.uint, label_slices: list, label_img = None, intensity_img = None) -> np.array:
    '''
    img: the whole image
    label_slices: [(node_label, slices),... ]
    label_img: slic label
    intensity_img: pred
    return: N_pieces * N_feats
    '''
    # modify label_img to avoid feat computing for regions that are not in label_slices
    needed_labels = [i[0] for i in label_slices]
    needed_label_img = label_img.copy()
    useless_labels = set(np.unique(label_img)) - set(needed_labels)
    useless_labels.remove(0)
    for useless_label in useless_labels: needed_label_img = np.where(needed_label_img == useless_label, 0, needed_label_img)
    
    # transform by whole image then extract patches
    img_hsv = img_as_ubyte(rgb2hsv(img))
    img_lab = rgb2lab(img) 
    img_lab = np.stack([img_lab[:,:,0] * 2.55, img_lab[:,:,1] + 128, img_lab[:,:,2] + 128], axis = -1).astype(np.uint)
    G_ce_ = equalize_adapthist(GrinsvenColorEnhance(img, pic_width = img.shape[1])[:,:,1])
    G_ce = img_as_ubyte(G_ce_)
    SCHARR = scharr(G_ce_)
    SCHARR = (SCHARR - SCHARR.min()) / (SCHARR.max() - SCHARR.min()).astype(np.float32)
    DOG = difference_of_gaussians(G_ce_, low_sigma = 0.3, high_sigma = 0.4)
    DOG = (np.clip(DOG, 0, None) / DOG.max()).astype(np.float32)
    PROPS = regionprops(label_img)
    # extract patches then transform
    output = []
    for label, slices in label_slices:
        patch_rgb = img[slices]
        patch_hsv = img_hsv[slices]
        patch_lab = img_lab[slices]
        patch_G_ce = G_ce[slices]
        patch_scharr = SCHARR[slices]
        patch_dog = DOG[slices]
        
        r_feats, _ = fos(patch_rgb[:,:,0], None)
        g_feats, _ = fos(patch_rgb[:,:,1], None)
        b_feats, _ = fos(patch_rgb[:,:,2], None)
        h_feats, _ = fos(patch_hsv[:,:,0], None)
        s_feats, _ = fos(patch_hsv[:,:,1], None)
        v_feats, _ = fos(patch_hsv[:,:,2], None)
        l_feats, _ = fos(patch_lab[:,:,0], None)
        a_feats, _ = fos(patch_lab[:,:,1], None)
        bb_feats, _ = fos(patch_lab[:,:,2], None)
        g_ce_feats, _ = fos(patch_G_ce, None)
        fos_feats = np.stack([r_feats, g_feats, b_feats, h_feats, s_feats, v_feats, l_feats, a_feats, bb_feats, g_ce_feats], axis = 0)
        fos_feats = fos_feats.reshape(fos_feats.size, )
        
        glcm_feats,_ ,_ ,_ = glcm_features(patch_G_ce, None) # only mean features of 4 directions
        glrlm_feats, _ = glrlm_features(patch_G_ce, None)
        glszm_feats, _ = glszm_features(patch_G_ce, None)
        
        scharr_mean, scharr_max, scharr_std = patch_scharr.mean(), patch_scharr.max(), patch_scharr.std()
        dog_mean, dog_max, dog_std = patch_dog.mean(), patch_dog.max(), patch_dog.std()
        edge_feats = np.array([scharr_mean, scharr_max, scharr_std, dog_mean, dog_max, dog_std])
        
        feats = np.concatenate((fos_feats, glcm_feats, glrlm_feats, glszm_feats, edge_feats), axis = 0)
        feats = np.reshape(feats, (1, feats.size))
        output.append(feats)
    feats = np.stack(output, axis = 1)[0,...]
    
    # get geo_feats
    needed_props = ['area', 'perimeter', 'solidity', 'eccentricity', 
                    'axis_major_length', 'axis_minor_length', 'orientation', 'bbox']
    
    props = regionprops_table(needed_label_img, intensity_img, properties = needed_props)
    props['circularity'] = 12.56 * props['area'] / props['perimeter'] ** 2
    props['aspect_ratio'] = (props['bbox-2'] - props['bbox-0']) / (props['bbox-3'] - props['bbox-1'])
    del props['bbox-0'], props['bbox-1'], props['bbox-2'], props['bbox-3']
    geo_feats = np.stack(list(props.values()))
    geo_feats = geo_feats.T
    
    feats = np.concatenate([feats, geo_feats], axis = -1)
    #print(f'hand feat extractor: {feats.shape[-1]} feats for every candidate')
    return feats
    

feature_flags = []

fos_flags = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
              "FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
              "FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
              "FOS_90Percentile","FOS_HistogramWidth"]


glcm_flags = ["GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
              "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
               "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
               "GLCM_Entropy", "GLCM_DifferenceVariance",
               "GLCM_DifferenceEntropy", "GLCM_Information1",
               "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"]

glrlm_flags = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_Short owGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]

glszm_flags = ['GLSZM_SmallZoneEmphasis', 'GLSZM_LargeZoneEmphasis',
              'GLSZM_GrayLevelNonuniformity', 'GLSZM_ZoneSizeNonuniformity',
              'GLSZM_ZonePercentage', 'GLSZM_LowGrayLeveLZoneEmphasis',
              'GLSZM_HighGrayLevelZoneEmphasis', 'GLSZM_SmallZoneLowGrayLevelEmphasis',
              'GLSZM_SmallZoneHighGrayLevelEmphasis', 'GLSZM_LargeZoneLowGrayLevelEmphassis',
              'GLSZM_LargeZoneHighGrayLevelEmphasis', 'GLSZM_GrayLevelVariance',
              'GLSZM_ZoneSizeVariance','GLSZM_ZoneSizeEntropy']

geo_flags = ['area', 'perimeter', 'solidity', 'eccentricity', 'axis_major_length', 
             'axis_minor_length', 'orientation', 'circularity', 'aspect_ratio']


for plane_id in ['r', 'g', 'b', 'h', 's', 'v', 'l', 'a', 'bb', 'g_ce']:
    feature_flags += [fos_flag + '_' +plane_id for fos_flag in fos_flags]
    
feature_flags += (glcm_flags + glrlm_flags + glszm_flags)

for plane_id in ['scharr', 'dog']:
    feature_flags += [plane_id + '_' + i for i in ['mean', 'max', 'std']]
    
feature_flags += geo_flags






























