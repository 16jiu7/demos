# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:03:09 2021

@author: 20598
"""
import numpy as np
import skimage.filters as filters
from skimage.util import img_as_ubyte

alpha = 4
beta = -4
_sigma = 512/30
gamma = 128

def GrinsvenColorEnhance(img: np.uint, pic_width: int = 512) -> np.float32:
    _sigma = pic_width / 30
    img = (img/255.).astype(np.float32)
    filtered = filters.gaussian(img, sigma = _sigma, channel_axis = True, preserve_range = True)
    out = alpha*img + beta*filtered + gamma
    return normalization(out)
    
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range









