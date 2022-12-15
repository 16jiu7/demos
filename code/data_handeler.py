#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 21:29:06 2022
handeling the 4 retinal image datasets
@author: jiu7
"""
import os
import skimage.io as io
from skimage.util import img_as_float32, img_as_ubyte
import numpy as np

# GLOBAL SETTINGS
DRD = os.path.join('/', 'home', 'jiu7', '4_retinal_datasets') # Dataset Root Dir
DRIVE_SPLIT = (15, 5, 20) # number of training, val and test images resp.
CHASEDB_SPLIT = (12, 2, 14)
HRF_SPLIT = (13, 2, 30)
STARE_SPLIT = (8, 2, 10) 

class single_data():
    # class to store everything about a single data "point"
    def __init__(self, ID, ori, fov_mask, gt, split):
        self.ID = ID
        self.ori = ori
        self.fov_mask = fov_mask
        self.gt = gt
        assert split in ['training', 'val','test']
        self.split = split
        self.pred = None
        self.shape = (None, None)

class RetinalDataset():
    # class whos instance is a retinal dataset like DRIVE = RetinalDataset(...)
    def __init__(self, name, visualize = False):
        assert name in ['DRIVE', 'CHASEDB', 'HRF', 'STARE'], 'name must be in DRIVE, CHASEDB, HRF, STARE, got f{name}'
        self.name = name
        self.all_data = self.get_data()
        self.all_training, self.all_val, self.all_test = self.split_data()
        self.visualize('./images/') if visualize else None
             
    def get_data(self):
        data_list = eval('get_' + self.name + '(self)')
        
        return data_list
    
    def split_data(self):
        all_training = [data for data in self.all_data if data.split == 'training']
        all_val = [data for data in self.all_data if data.split == 'val']
        all_test = [data for data in self.all_data if data.split == 'test']
        print(f'for {self.name}, {len(self.all_data)} in total, {len(all_training)} for training, {len(all_val)} for val and {len(all_test)} for test')
        return all_training, all_val, all_test
    
    def visualize(self, save_dir):
        for data in self.all_data:
            gt = np.stack([data.gt]*3, axis = -1) if data.gt.ndim == 2 else data.gt
            fov_mask = np.stack([data.fov_mask]*3, axis = -1) if data.fov_mask.ndim == 2 else data.fov_mask
            ori = data.ori
            image = np.concatenate([ori, gt, fov_mask], axis = 0)
            io.imsave(os.path.join(save_dir, f'{self.name}_{data.ID}_{data.split}.png'), img_as_ubyte(image))
    
def get_DRIVE(self):
    all_DRIVE = []
    img_names = os.listdir(os.path.join(DRD, 'DRIVE', 'images'))
    img_names = sorted(img_names, key = lambda x : x.split(os.sep)[-1].split('_')[0])
    img_names_test, img_names_training, img_names_val = img_names[:20], img_names[20:35], img_names[35:]
    
    for split in ['training', 'val', 'test']:
        for idx, img_name in enumerate(eval('img_names_' + split)):
            ID = img_name.split('_')[0] # '01' '02' and so on
            ori = io.imread(os.path.join(DRD, 'DRIVE', 'images', img_name))
            fov_mask = io.imread(os.path.join(DRD, 'DRIVE', 'masks', f'{ID}_mask.gif'))
            gt = io.imread(os.path.join(DRD, 'DRIVE', 'manual', f'{ID}_manual1.gif'))
            drive_instance = single_data(ID, ori, fov_mask, gt, split)
            all_DRIVE.append(drive_instance)
            
    return all_DRIVE    
        
def get_CHASEDB(self):
    all_CHASEDB = []
    img_names = os.listdir(os.path.join(DRD, 'CHASEDB', 'images'))
    img_names = sorted(img_names)
    img_names_training, img_names_val, img_names_test = \
    img_names[:CHASEDB_SPLIT[0]], img_names[CHASEDB_SPLIT[0]:CHASEDB_SPLIT[0] + CHASEDB_SPLIT[1]], img_names[CHASEDB_SPLIT[0] + CHASEDB_SPLIT[1]:]
    for split in ['training', 'val', 'test']:
        for idx, img_name in enumerate(eval('img_names_' + split)):
            ID = img_name.split('.')[0].split('_')[-1] # somthing like 01L, 01R etc.
            ori = io.imread(os.path.join(DRD, 'CHASEDB', 'images', img_name))
            fov_mask = io.imread(os.path.join(DRD, 'CHASEDB', 'masks', f'Image_{ID}.gif'))
            gt = io.imread(os.path.join(DRD, 'CHASEDB', 'manual', f'Image_{ID}_1stHO.png'))
            chasedb_instance = single_data(ID, ori, fov_mask, gt, split)
            all_CHASEDB.append(chasedb_instance)
        
    return all_CHASEDB      

def get_HRF(self):
    all_HRF = []
    img_names = os.listdir(os.path.join(DRD, 'HRF', 'images'))
    img_names = sorted(img_names)
    img_names_training, img_names_val, img_names_test = \
    img_names[:HRF_SPLIT[0]], img_names[HRF_SPLIT[0]:HRF_SPLIT[0] + HRF_SPLIT[1]], img_names[HRF_SPLIT[0] + HRF_SPLIT[1]:]
    
    for split in ['training', 'val', 'test']:
    
        for idx, img_name in enumerate(eval('img_names_' + split)):
            ID = img_name.split('.')[0].split('_')[-1] # somthing like 01, 02 etc.
            ori = io.imread(os.path.join(DRD, 'HRF', 'images', img_name))
            fov_mask = io.imread(os.path.join(DRD, 'HRF', 'masks', f'{ID}.png'))
            gt = io.imread(os.path.join(DRD, 'HRF', 'manual1', f'{ID}.png'))
            HRF_instance = single_data(ID, ori, fov_mask, gt, split)
            all_HRF.append(HRF_instance) 
        
    return all_HRF  
def get_STARE(self):
    all_STARE = []
    img_names = os.listdir(os.path.join(DRD, 'STARE', 'images'))
    img_names = sorted(img_names)
    img_names_training, img_names_val, img_names_test = \
    img_names[:STARE_SPLIT[0]], img_names[STARE_SPLIT[0]:STARE_SPLIT[0] + STARE_SPLIT[1]], img_names[STARE_SPLIT[0] + STARE_SPLIT[1]:]
    for split in ['training', 'val', 'test']:
        for idx, img_name in enumerate(eval('img_names_' + split)):
            ID = img_name.split('.')[0] # somthing like im0001 etc.
            ori = io.imread(os.path.join(DRD, 'STARE', 'images', img_name))
            fov_mask = io.imread(os.path.join(DRD, 'STARE', 'masks', f'{ID}.gif'))
            gt = io.imread(os.path.join(DRD, 'STARE', 'labels-ah', f'{ID}.ah.pgm'))
            all_STARE.append(single_data(ID, ori, fov_mask, gt, split))
        
    return all_STARE         
    
    
# if __name__ == '__main__':
#     drive = RetinalDataset('DRIVE')
#     chasedb = RetinalDataset('CHASEDB')
#     hrf = RetinalDataset('HRF')
#     stare = RetinalDataset('STARE')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    