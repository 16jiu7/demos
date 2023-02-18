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
from copy import deepcopy
from data_handeler import RetinalDataset
from torch.utils.data import Dataset, DataLoader
from models.unet_gat import UNet_3_32
from torchmetrics.classification import BinaryAUROC, Accuracy, Precision, Recall, Specificity, F1Score
from torchvision.transforms import ToTensor
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
import albumentations as A
from torch.nn import BCEWithLogitsLoss
totensor = ToTensor()

def setup_random_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_test_resizer(dataset_name):    
    test_resizer = A.Compose([Resize(*INPUT_SIZE[dataset_name])]) 
    return test_resizer

class TrainDataset(Dataset):
    def __init__(self, dataset_name, split):
        self.split = split
        self.dataset_name = dataset_name
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
            # give img that was cropped to fov and resized to INPUT_SIZE with original gt, bbox to enable resized prediction 
            cropped_img, gt = self.data[idx].cropped_ori, self.data[idx].gt
            test_resizer = get_test_resizer(self.dataset_name)
            transformed = test_resizer(image = cropped_img)
            resized_cropped_img = transformed['image']
            return totensor(resized_cropped_img), totensor(gt), self.data[idx].bbox, self.data[idx].ID
        
        if self.split in ['train', 'val']:
            img, gt = self.data[idx].ori, self.data[idx].gt
            transformed = transforms(image = img, mask = gt)
            img, gt = transformed['image'], transformed['mask']
            return totensor(img), totensor(gt) 
    


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")

class StepRunner:
    def __init__(self, net, loss_fn,
                 stage = "train", metrics_dict = None, 
                 optimizer = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer = optimizer
            
    def step(self, features, labels):
        #loss
        preds = self.net(features)
        preds = preds[:,0,:,:].unsqueeze(1)
        loss = self.loss_fn(preds,labels)
        
        #backward()
        if self.optimizer is not None and self.stage=="train": 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        #metrics
        step_metrics = {self.stage+"_"+name:metric_fn(preds, labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        return loss.item(),step_metrics
    
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
                loss_fn = loss_fn,metrics_dict = deepcopy(metrics_dict),
                optimizer = optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(net = net,stage="val",
                loss_fn = loss_fn,metrics_dict=deepcopy(metrics_dict))
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
    INPUT_SIZE = {'DRIVE': [512, 512], 'CHASEDB':[1024, 1024], 'HRF':[1024, 1024], 'STARE':[1024, 1024]}
    # INPUT_SIZE also used in test: cropped imgs are resized to INPUT_SIZE, then the prediction results
    # are upsampled and expanded to match gt
    for TRAIN_DATASET in TRAIN_DATASETS:
    
        SEED = 2023
        setup_random_seed(SEED)
        
        # data augmentations
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        transforms = A.Compose([Resize(*INPUT_SIZE[TRAIN_DATASET]), RandomRotate90(p = 0.5), Flip(p = 0.5),
                                ColorJitter(brightness, contrast, saturation, hue, always_apply = True)])

        train_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'train')
        val_set = TrainDataset(dataset_name = TRAIN_DATASET, split = 'val')
    
        train_loader = DataLoader(train_set, batch_size = 1, num_workers = 4)
        val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4)
    
    
        net = UNet_3_32(3, 2)
    
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer= torch.optim.Adam(net.parameters(),lr = 1e-3)   
        auroc = BinaryAUROC(thresholds=None)
        metrics_dict = {"AUROC": auroc}
        
        dfhistory = train_model(net,
            optimizer,
            loss_fn,
            metrics_dict,
            train_data = train_loader,
            val_data= val_loader,
            epochs = 100,
            ckpt_path = f'../weights/pre_training/{TRAIN_DATASET}_pre.pt',
            patience = 30,
            monitor = "val_AUROC", 
            mode = "max")

# In[]

    # test for pre-training
    TEST_THRESH = 0.5 # the theshold used when calculating f1, acc, precision, recall, spe
    TEST_DATASETS = ['DRIVE', 'CHASEDB', 'HRF', 'STARE']
    PREDS_SAVE_DIR = '../preds/pre_training/' # preds saved here
    # define metrics
    auroc = BinaryAUROC(thresholds=None)
    acc = Accuracy(task = 'binary', threshold = TEST_THRESH)
    precision = Precision(task = 'binary', threshold = TEST_THRESH) 
    recall = Recall(task = 'binary', threshold = TEST_THRESH) 
    spe = Specificity(task = 'binary', threshold = TEST_THRESH) 
    f1 = F1Score(task = 'binary', threshold = TEST_THRESH)
    test_metrics = {'AUROC': auroc, 'f1': f1, 'acc': acc, 'pricision': precision, 'recall': recall, 'spe':spe}
    
    def test_1_pred(pred: torch.Tensor, gt: torch.Tensor, metrics: dict) -> list:
        # get every metric for 1 prediction, inputs are 2D tensors
        results = []
        for key in test_metrics.keys():
            metric = test_metrics[key]
            res = metric(pred, gt).item()
            results.append(res)
        return results
    
    
    net = UNet_3_32(3, 2)
    
    all_results = {}
    
    for DATASET_NAME in TEST_DATASETS:
        print(f'testing {DATASET_NAME}')
        
        if not os.path.exists(PREDS_SAVE_DIR + DATASET_NAME + '/'):
            os.mkdir(PREDS_SAVE_DIR + DATASET_NAME + '/')
        SAVE_PATH = PREDS_SAVE_DIR + DATASET_NAME + '/'
        
        checkpoint = torch.load(f'../weights/pre_training/{DATASET_NAME}_pre.pt')
        net.load_state_dict(checkpoint)
        #net.cuda()
        net.eval()
        
        test_set = TrainDataset(dataset_name = DATASET_NAME, split = 'test')
        test_loader = DataLoader(test_set, batch_size = 1, num_workers = 4)
        
        results_for_1_dataset = []
        with torch.no_grad():
            # get pred vessel map
            for resized_cropped_img, gt, bbox, ID in test_loader:
                bbox_resizer = nn.Upsample(size = (bbox[2] - bbox[0], bbox[3] - bbox[1]), mode='bilinear', align_corners=True)
                
                inputs, gt = resized_cropped_img.cpu(), gt.cpu()
                pred = nn.functional.softmax(net(inputs), dim=1)

                # refine pred so that it has same shape like gt  
                pred = bbox_resizer(pred)[0,0]
                gt = gt[0,0]
                tmp = torch.zeros_like(gt)
                tmp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = pred
                refined_pred = tmp
                io.imsave(SAVE_PATH + ID[0] + '.png', img_as_ubyte(refined_pred.numpy()))
                
                results_for_1 = test_1_pred(refined_pred, gt, test_metrics)
                results_for_1_dataset.append(results_for_1)
            results_for_1_dataset = np.array(results_for_1_dataset)
        
        all_results[DATASET_NAME] = results_for_1_dataset
        
    import pickle
    with open('with_colorjitter.pkl', "wb") as tf:
        pickle.dump(all_results, tf)

# In[]

    # read results from previous experiments
    with open("results_for_all.pkl", "rb") as tf:
        results_for_all = pickle.load(tf)









