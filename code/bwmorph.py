# Duplication of 'bwmorph' in matlab
# referred to
# https://gist.github.com/joefutrelle/562f25bbcf20691217b8


import numpy as np
from scipy import ndimage as ndi


OPS = ['dilate', 'fill', 'thin', 'branchpoints', 'endpoints']


# lookup tables
LUT_THIN_1 = ~np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,0,0,0,1,0,0,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=bool)

LUT_THIN_2 = ~np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1], dtype=bool)

LUT_ENDPOINTS = ~np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,0], dtype=bool)

LUT_BRANCHPOINTS = ~np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=bool)

LUT_BACKCOUNT4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,1,1,1,1,2,1,1,1,1,2,1,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,1,2,1,1,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,1,1,2,1,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           2,3,3,3,2,3,2,2,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,2,2,3,2,3,3,3,2,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           2,3,3,3,2,3,2,2,3,3,4,3,3,3,3,2,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,1,1,2,1,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           2,3,3,3,2,3,2,2,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,2,2,3,2,3,3,3,2,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,1,2,1,1,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           2,3,3,3,3,4,3,3,2,2,3,2,3,3,3,2,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           2,3,3,3,2,3,2,2,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,2,2,3,2,3,3,3,2,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,1,2,1,1,2,2,3,2,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,2,3,2,2,1,1,2,1,2,2,2,1,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           1,2,2,2,1,2,1,1,1,1,2,1,1,1,1,0], dtype=np.uint8)

LUT_DILATE = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=bool)

LUT_FILL = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,
                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=bool)


def bwmorph(image, op, n_iter=None):

    # check parameters
    if op not in OPS:
        raise ValueError('Undefined OP is used')    
    
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    bw = np.array(image).astype(np.uint16)
    
    if bw.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[  1,  8, 64],
                     [  2, 16,128],
                     [  4, 32,256]],dtype=np.uint16)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(bw) # count points before thinning
        
        if op == 'dilate':
           
            bw[np.take(LUT_DILATE, ndi.correlate(bw, mask, mode='constant'))] = 1
            
        elif op == 'fill':
            
            bw[np.take(LUT_FILL, ndi.correlate(bw, mask, mode='constant'))] = 1
        
        elif op == 'thin':
            
            # for each subiteration
            for lut in [LUT_THIN_1, LUT_THIN_2]:
                # correlate image with neighborhood mask
                N = ndi.correlate(bw, mask, mode='constant')
                # take deletion decision from this subiteration's LUT
                D = np.take(lut, N)
                # perform deletion
                bw[D] = 0
            
        elif op == 'branchpoints':
            
            # Initial branch point candidates
            C = np.copy(bw)
            C[np.take(LUT_BRANCHPOINTS, ndi.correlate(bw, mask, mode='constant'))] = 0
            C = C.astype(np.bool)
            
            # Background 4-Connected Object Count (Vp)            
            B = np.take(LUT_BACKCOUNT4, ndi.correlate(bw, mask, mode='constant'))
            
            # End Points (Vp = 1)
            E = (B == 1)
            
            # Final branch point candidates
            F = (~E)*C
            
            # Generate mask that defines pixels for which Vp = 2 and no
            # foreground neighbor q for which Vq > 2
            
            # Vp = 2 Mask
            Vp = ((B == 2) & (~E))
            
            # Vq > 2 Mask
            Vq = ((B > 2) & (~E))
            
            # Dilate Vq
            D = np.copy(Vq)
            D[np.take(LUT_DILATE, ndi.correlate(Vq, mask, mode='constant'))] = 1
            
            # Intersection between dilated Vq and final candidates w/ Vp = 2
            M = (F & Vp) & D
            
            # Final Branch Points
            bw =  F & (~M)
            
            break
            
        elif op == 'endpoints':
            
            # correlate image with neighborhood mask
            N = ndi.correlate(bw, mask, mode='constant')
            # take deletion decision from the LUT
            D = np.take(LUT_ENDPOINTS, N)
            # perform deletion
            bw[D] = 0
            
        else:
            pass
        
        after = np.sum(bw) # count points after thinning
        
        if before == after:
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    
    return bw.astype(np.bool)

