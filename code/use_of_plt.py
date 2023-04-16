#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:40:08 2023

@author: jiu7
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 150, num = 150)

y = np.random.randn(150)
y = y.tolist()
plt.figure(figsize=(10,5), dpi = 200)
plt.subplot(211)
plt.title('loss')
plt.grid()
plt.plot(x,y, c = 'b', marker = '.', linewidth = 1, markersize = 2)


plt.subplot(212)
plt.title('val')
plt.grid()
plt.plot(x,y, c = 'r', marker = '.', linewidth = 1, markersize = 2)

plt.tight_layout() # otherwise subplots will land on each other
plt.savefig('random.png')


def draw_figs(x, losses, vals):
    plt.figure(figsize=(10,5), dpi = 200)
    plt.subplot(211)
    plt.title('loss')
    plt.grid()
    plt.plot(x,y, c = 'b', marker = '.', linewidth = 1, markersize = 2)

    plt.subplot(212)
    plt.title('val')
    plt.grid()
    plt.plot(x,y, c = 'r', marker = '.', linewidth = 1, markersize = 2)

    plt.tight_layout() # otherwise subplots will land on each other
    plt.savefig('random.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    