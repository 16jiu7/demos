""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GAT import GAT
import numpy as np
from torchvision.transforms import ToTensor
from skimage.measure import regionprops
from make_graph_light import GraphedImage
from data_handeler import RetinalDataset
import networkx as nx
import datetime

DIM_ENCODER_FEATS = 256 # k
GAT_INPUT_N_FEATS = 256
GAT_MID_N_C = 256
GAT_OUTPUT_N_FEATS = 256

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_3_32(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, with_feat = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_feat = with_feat

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down3 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        if self.with_feat:
            resizer = nn.Upsample(size = x1.shape, mode='bilinear', align_corners=True)
            feats = torch.cat([x1, resizer(x2), resizer(x3)], dim = 1)
            print(feats.size())
            return logits, feats
        else:
            return logits

    def get_concat_feats(self, x):
        # only get downscaled feats, without CNN pred result
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        resizer1 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        resizer2 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        feats = torch.cat([x1, resizer1(x2), resizer2(x3)], dim = 1)
        return feats
    
    def get_encoder_feats(self, x):
        # only get downscaled feats, without CNN pred result
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4 # shape = N, 256, H/8, W/8

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
        
    
def GetCNNFeats(img, net):
    # only get CNN feats for GNN use
    # feats.shape = NkHW
    img = img.astype(np.float32)
    img = ToTensor()(img).unsqueeze(0)
    feats = net.get_encoder_feats(img)
    return feats

def GetNodeFeats(cnn_feats, graphedpred):
    # cnn_feats: NkHW, k = 144
    # graphedpred: GraphedImage instance
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graphedpred.graph.nodes)
    node_feats = torch.zeros(size = [len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N_nodes, k
    slic_label = graphedpred.slic_label
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        feat_slices = [slice(None), slice(None), r_props[node - 1].slice[0], r_props[node - 1].slice[1]]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[1], -1) # k,H_bbox*W_bbox
        _, _, V = torch.pca_lowrank(node_feats_i, center = True)
        node_feat_i = torch.matmul(node_feats_i, V[:, :1]) # k,1
        node_feats[i, :] = node_feat_i[:, 0]
        
    return node_feats

def GetNodeFeatsFromSmall(cnn_feats, graphedpred, down_ratio = 8):
    # cnn_feats: N, k, H/n, W/n, k = 256, n = down scale factor
    # graphedpred: GraphedImage instance
    # order: by labels in graph.nodes
    # assume batch_size = 1
    nodes = list(graphedpred.graph.nodes)
    node_feats = torch.zeros(size = [len(nodes), cnn_feats.shape[1]], dtype = torch.float32) # N_nodes, k
    slic_label = graphedpred.slic_label
    r_props = regionprops(slic_label)
    
    for i, node in enumerate(nodes):
        min_row, min_col, max_row, max_col = r_props[node - 1].bbox
        min_row, min_col, max_row, max_col = \
        min_row // down_ratio, min_col // down_ratio, max_row // down_ratio, max_col // down_ratio
        feat_slices = [slice(None), slice(None), slice(min_row, max_row), slice(min_col, max_col)]
        node_feats_i = cnn_feats[feat_slices] # all feats within bbox, shape = N,k,H_bbox,W_bbox
        node_feats_i = node_feats_i.reshape(node_feats_i.shape[1], -1) # k,H_bbox*W_bbox
        if node_feats_i.shape[-1] > 1:
            _, _, V = torch.pca_lowrank(node_feats_i, center = True)
            node_feat_i = torch.matmul(node_feats_i, V[:, :1]) # k,1
        node_feats[i, :] = node_feat_i[:, 0]
        
    return node_feats

def FuseFeats(cnn_feats, node_feats, graph, down_ratio = 8, method = 'replace'):
    # fuse node feats(GAT output) into CNN feats
    # graph can be graphedpred.graph after relabeling
    # scale factor = cnn feat size / ori size
    if method == 'replace':
        for node in graph.nodes:
            pos = (graph.nodes[node]['center'][0] // down_ratio, graph.nodes[node]['center'][1] // down_ratio)
            cnn_feats[:, :, pos[0], pos[1]] = node_feats[node, :]
    
    return cnn_feats
            
        
        
if __name__ == '__main__':
    # what kind of CNN feats to extract node feats from:
    # 1.plain encoder feats, shape = N, 256, H/8, W/8
    # 2.deep multi-scale feats, shape = N, 256+32+64+128, H/8, W/8
    # 3.large multi-scale feats, shape = N, 256+32+64+128, H, W
    
    # how to compress node feats
    # 1.max pooing
    # 2.PCA
    starttime = datetime.datetime.now()

    data = RetinalDataset('DRIVE').all_training[0] # the first training img in DRIVE
    graphedpred = GraphedImage(data.pred, data.fov_mask, 1500)
    # define net
    net = UNet_3_32(3, 2)  
    checkpoint = torch.load('../weights/UNet_3_32.pt7')
    net.load_state_dict(checkpoint['net'])        
        
    cnn_feats = GetCNNFeats(data.ori, net)
    node_feats = GetNodeFeatsFromSmall(cnn_feats, graphedpred)
    # once we got node_feats, we can relabel the graph 
    mapping = {}
    old_labels = list(graphedpred.graph.nodes)
    new_labels = [i for i in range(len(old_labels))] # new node labels start from 0, due to GAT need
    for i, old_label in enumerate(old_labels): mapping[old_label] = new_labels[i]
    relabeled_graph = nx.relabel_nodes(graphedpred.graph, mapping)
    edges = list(relabeled_graph.edges)
    edges = torch.Tensor(edges).long().transpose(1,0) # shape = (2, E), data type = torch.long for GAT use
    graph_data = (node_feats, edges)
    
    gat = GAT(num_of_layers = 4, num_heads_per_layer = [4,4,4,4], 
               num_features_per_layer = [GAT_INPUT_N_FEATS, GAT_MID_N_C, GAT_MID_N_C, GAT_MID_N_C, GAT_OUTPUT_N_FEATS], dropout = 0)
    
    node_feats_gat, _ = gat(graph_data)
    fused_feats = FuseFeats(cnn_feats, node_feats_gat, relabeled_graph)
    
    endtime = datetime.datetime.now()
    print(f'Run time : {endtime - starttime}s')
    
    
    
    # def GetFirstPCNN(img, net):
    #     # get temp CNN pred result, used in node sampling & graph building
    #     # net in no_gnn mode
    #     # only run once when training begins
        
        
    #     pass




    # def ReProjectNodeFeats(node_feats: list, slic_label, scale_factor):
    #     # place node features (the GAT outputs) into the downscaled feature map, 
    #     # if downscaled slic label no longer have certain pieces, just ignore them
    #     N_nodes, N_dim = node_feats.shape[0], node_feats.shape[1]
    #     resized_label = resize(label, output_shape = \
    #     (int(584 / scale_factor), int(565 / scale_factor)), preserve_range=True).astype(np.int64)
    #     survived_labels = list(np.unique(resized_label))
    #     survived_labels.remove(0)
        
    #     props = regionprops(label_image = resized_label, intensity_image = None)
    #     for prop in props:
    #         pass


    # def GetFinRes():
    #     # get the final CNN+GAT pred result
    #     pass

    









     
