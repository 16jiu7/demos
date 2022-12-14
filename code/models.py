#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:36:29 2022
define GAT and CNN models
define the mixed infer model
@author: jiu7
"""
import torch
import networkx as nx
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
import numpy as np

