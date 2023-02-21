'''
perform pre-training, to produce the initial pred results for further training
method: training on whole images, batch_size = 1, on each dataset
pre-traning is performed without GNN
'''
import skimage.io as io
from skimage.util import img_as_ubyte
import os,sys,time
import numpy as np
import pandas as pd
import datetime 
from tqdm import tqdm 
import random
import torch
from torch import nn 
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
from copy import deepcopy
from data_handeler import RetinalDataset
from torch.utils.data import Dataset, DataLoader
from models.unet_gat import UNet_3_32
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC, Accuracy, Precision, Recall, Specificity, F1Score
from torchvision.transforms import ToTensor
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip, Affine
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
import albumentations as A
totensor = ToTensor()

USE_AMP = False

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split, transforms = None):
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        if split == 'train':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_training
        elif split == 'val':
            self.data = RetinalDataset(self.dataset_name, cropped = True).all_val
        elif split == 'test':
            self.data = RetinalDataset(self.dataset_name, cropped = False).all_test # metrics should be calculate on original images
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.split == 'test':
            cropped_img, gt = self.data[idx].cropped_ori, self.data[idx].gt # give cropped ori & non-cropped gt
            return totensor(cropped_img).cuda(), totensor(gt).cuda(), self.data[idx].bbox, self.data[idx].ID
        
        if self.split in ['train', 'val']:
            img, gt = self.data[idx].ori, self.data[idx].gt
            if self.transforms:
                transformed = self.transforms(image = img, mask = gt)
                img, gt = transformed['image'], transformed['mask']
            return totensor(img).cuda(), totensor(gt).cuda()
    


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")

class StepRunner:
    def __init__(self, net, Loss_fn,
                 stage = "train", metrics_dict = None, 
                 optimizer = None
                 ):
        self.net,self.Loss_fn,self.metrics_dict,self.stage = net,Loss_fn,metrics_dict,stage
        self.optimizer = optimizer
            
    def step(self, features, labels):
        with torch.no_grad():
            weight = torch.zeros(2, dtype = torch.float)
            beta = (labels.sum() / (labels.shape[2] * labels.shape[3])).float() # beta < 0.5
            weight[0], weight[1] = beta, 1 - beta
            print(weight)
            
        loss_fn = self.Loss_fn(weight = weight.cuda()) # instantiate
        
        if USE_AMP:
            with autocast():
                preds = self.net(features)
                loss =loss_fn(preds,labels.squeeze(1).long())
        else:
            preds = self.net(features)
            loss = loss_fn(preds,labels.squeeze(1).long())

        #backward()
        if self.optimizer is not None and self.stage=="train": 
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        #metrics
        step_metrics = {self.stage+"_"+name:metric_fn(preds[:,1,:,:], labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics
    
    def train_step(self,features,labels):
        self.net.train() #训练模式, dropout层发生作用
        return self.step(features,labels)
    
    @torch.no_grad()
    def eval_step(self,features,labels):
        self.net.eval() #预测模式, dropout层不发生作用
        return self.step(features,labels)
    
    def __call__(self,features,labels):
        if self.stage=="train":
            return self.train_step(features,labels) 
        else:
            return self.eval_step(features,labels)
        
class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        
    def __call__(self, dataloader):
        total_loss,step = 0,0
        loop = tqdm(enumerate(dataloader), total =len(dataloader), file = sys.stdout)
        for i, batch in loop: 
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage+"_loss":loss},**step_metrics)
            total_loss += loss
            step+=1
            if i!=len(dataloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {self.stage+"_"+name:metric_fn.compute().item() 
                                  for name,metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage+"_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, optimizer, loss_fn, metrics_dict, 
                train_data, val_data=None, 
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min"):
    
    history = {}

    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------  
        train_step_runner = StepRunner(net = net,stage="train",
                Loss_fn = loss_fn,metrics_dict = deepcopy(metrics_dict),
                optimizer = optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(net = net,stage="val",
                Loss_fn = loss_fn,metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:
            torch.save(net.state_dict(),ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                 arr_scores[best_score_idx]))
        if len(arr_scores)-best_score_idx>patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor,patience))
            break 
    net.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)

def get_net(net): #获得预训练模型并冻住前面层的参数
    '''Freeze all layers except the last layer(fc or classifier)'''
    for param in net.parameters():
        param.requires_grad = False
    # nn.init.xavier_normal_(model.fc.weight)
    # nn.init.zeros_(model.fc.bias)
    net.fc.weight.requires_grad = True
    net.fc.bias.requires_grad = True
    return net

# In[]

if __name__ == '__main__':
    
    TRAIN_DATASETS = ['DRIVE', 'CHASEDB', 'HRF', 'STARE']    
    TRAIN_DATASETS = ['CHASEDB']    
    TRAIN_INPUT_SIZE = {'DRIVE': [584, 565], 'CHASEDB':[960, 999], 'HRF':[2336, 3504], 'STARE':[605, 700]}
    # INPUT_SIZE also used in test: cropped imgs are resized to INPUT_SIZE, then the prediction results
    # are upsampled and expanded to match gt
    for TRAIN_DATASET in TRAIN_DATASETS:
    
        SEED = 2023
        setup_random_seed(SEED)
        
        # data augmentations
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        transforms = A.Compose([Resize(*TRAIN_INPUT_SIZE[TRAIN_DATASET]), RandomRotate90(p = 0.5), Flip(p = 0.5),
                                Affine(scale = (0.95, 1.2), translate_percent = 0.05, rotate = (-45, 45)),
                                ColorJitter(brightness, contrast, saturation, hue, always_apply = True)])
        val_resizer = A.Compose([Resize(*TRAIN_INPUT_SIZE[TRAIN_DATASET])])
        
        # colorjitter severely affects performance
        if TRAIN_DATASET in ['DRIVE', 'CHASEDB', 'STARE']:
            train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train', transforms = transforms)
            val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val', transforms = None)
            
        elif TRAIN_DATASET == 'HRF': # HRF images are too large, need resizer
            train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train', transforms = transforms)
            val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val', transforms = val_resizer)
            
        train_loader = DataLoader(train_set, batch_size = 1, num_workers = 0)
        val_loader = DataLoader(val_set, batch_size = 1, num_workers = 0)
    
    
        net = UNet_3_32(3, 2).cuda()
    
        loss_fn = nn.CrossEntropyLoss
        optimizer= torch.optim.Adam(net.parameters(),lr = 1e-2, weight_decay = 5e-4)   
        auroc = BinaryAUROC(thresholds=None).cuda()
        metrics_dict = {"AUROC": auroc}
        
        dfhistory = train_model(net,
            optimizer,
            loss_fn,
            metrics_dict,            
            train_data = train_loader,
            val_data= val_loader,
            epochs = 5000,
            ckpt_path = f'../weights/pre_training/{TRAIN_DATASET}_pre.pt',
            patience = 1000,
            monitor = "val_AUROC", 
            mode = "max")

        torch.cuda.empty_cache()
# In[]

    # test for pre-training
    TEST_THRESH = 0.5 # the theshold used when calculating f1, acc, precision, recall, spe
    TEST_INPUT_SIZE = {'DRIVE': [584, 565], 'CHASEDB':[960, 999], 'HRF':[1024, 1552], 'STARE':[605, 700]}
    #assert TEST_INPUT_SIZE == TRAIN_INPUT_SIZE
    TEST_DATASETS = ['DRIVE', 'CHASEDB', 'HRF', 'STARE']
    TEST_DATASETS = ['CHASEDB']
    PREDS_SAVE_DIR = '../preds/pre_training/' # preds saved here
    # define metrics
    ap = BinaryAveragePrecision(thresholds=None).to('cuda')
    auroc = BinaryAUROC(thresholds=None).to('cuda')
    f1 = F1Score(task = 'binary', threshold = TEST_THRESH).to('cuda')
    acc = Accuracy(task = 'binary', threshold = TEST_THRESH).to('cuda')
    precision = Precision(task = 'binary', threshold = TEST_THRESH).to('cuda')
    recall = Recall(task = 'binary', threshold = TEST_THRESH).to('cuda')
    spe = Specificity(task = 'binary', threshold = TEST_THRESH).to('cuda')
    test_metrics = {'AP': ap, 'AUROC': auroc, 'f1': f1, 'acc': acc, 'pricision': precision, 'recall': recall, 'spe':spe}
    
    def test_1_pred(pred: torch.Tensor, gt: torch.Tensor, metrics: dict) -> list:
        # get every metric for 1 prediction, inputs are 2D tensors
        results = []
        for key in test_metrics.keys():
            metric = test_metrics[key]
            res = metric(pred, gt).item()
            results.append(res)
        return results
    
    
    net = UNet_3_32(3, 2).cuda()
    
    all_results = {}
    
    for DATASET_NAME in TEST_DATASETS:
        print(f'testing {DATASET_NAME}')
        
        if not os.path.exists(PREDS_SAVE_DIR + DATASET_NAME + '/'):
            os.mkdir(PREDS_SAVE_DIR + DATASET_NAME + '/')
        SAVE_PATH = PREDS_SAVE_DIR + DATASET_NAME + '/'
        
        checkpoint = torch.load(f'../weights/pre_training/{DATASET_NAME}_pre.pt')
        net.load_state_dict(checkpoint)
        net.eval()
        
        test_set = TrainDataset(dataset_name = DATASET_NAME, split = 'test')
        test_loader = DataLoader(test_set, batch_size = 1, num_workers = 0)
        
        results_for_1_dataset = []
        with torch.no_grad():
            # get pred vessel map
            for cropped_img, gt, bbox, ID in test_loader:
                
                if DATASET_NAME == 'HRF':
                    bbox_resizer = nn.Upsample(size = (bbox[2] - bbox[0], bbox[3] - bbox[1]), mode='bilinear', align_corners=True)
                    cropped_img = bbox_resizer(cropped_img)
                    
                inputs, gt = cropped_img.cuda(), gt.cuda()
                
                pred = nn.functional.softmax(net(cropped_img), dim=1)

                # refine pred so that it has same shape like gt  
                pred = pred[0,1]
                gt = gt[0,0]
                tmp = torch.zeros_like(gt)
                tmp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = pred
                refined_pred = tmp
                io.imsave(SAVE_PATH + ID[0] + '.png', img_as_ubyte(refined_pred.cpu().numpy()))
                
                results_for_1 = test_1_pred(refined_pred, gt, test_metrics)
                results_for_1_dataset.append(results_for_1)
            results_for_1_dataset = np.array(results_for_1_dataset)
        
        all_results[DATASET_NAME] = results_for_1_dataset
        
    import pickle
    with open('../preds/with_colorjitter_geo.pkl', "wb") as tf:
        pickle.dump(all_results, tf)
    
    means = {}
    for key in all_results.keys():
        means[key] = all_results[key].mean(axis = 0)
    with open('../preds/performances.txt', 'w') as f:
        for key in all_results.keys():
            f.writelines(key + '\n')

# In[]

import pickle

with open('../preds/performances/with_colorjitter.pkl', "rb") as tf:
    all_results = pickle.load(tf)

with open('../preds/performances/with_colorjitter_geo.pkl', "rb") as tf:
    all_results_geo = pickle.load(tf)



