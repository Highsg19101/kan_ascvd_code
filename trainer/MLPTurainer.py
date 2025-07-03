# Tuning & Training = Turainer

import torch
import torch.nn as nn
from models.mlp import MLP
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import numpy as np
from utils.KFolds_splitting import *

import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import yaml
from sklearn.impute import KNNImputer
import torch.optim as optim
from utils.Metric import *
from utils.Loader_for_mlp import *
from models.mlp import MLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


def class_weight(train):
    pos = train[train['ASCVD']==1]
    neg = train[train['ASCVD']==0]
    
    class_counts = [neg.shape[0],pos.shape[0]]
    total_samples = sum(class_counts)
    weights = [total_samples / count for count in class_counts]
    print("Pos:",pos.shape[0],"Neg:",neg.shape[0],":",weights)
    weight_sum = sum(weights)
    weights = [w/ weight_sum for w in weights]
    weights = torch.tensor(weights, dtype=torch.torch.float32)
    print("Normalized Class Weights:",weights)
    return weights


class MLPTuner:
    def __init__(self,
                 train_df : pd.DataFrame,
                 valid_df : pd.DataFrame,
                 seed : int,
                 var_mode: str,
                 device: str,
                 batch: int,
                 path_guide
                 ):
        self.train_df = train_df
        self.valid_df = valid_df
        self.var_mode = var_mode
        self.seed = seed
        self.device = device
        self.scaler = StandardScaler()
        set_seed(self.seed)
        self.batch = batch
        self.path_guide = path_guide

    def hype_search(self):
        with open(self.path_guide.cfg_path, 'r') as file:
            config = yaml.safe_load(file)
        if self.var_mode == 'PRIMARY':
            MLPelements = config['MLPPrimaryTuneElement']
        elif self.var_mode == 'EXTEND':
            MLPelements = config['MLPExtendTuneElement']
        elif self.var_mode == 'EXTEND_Toy':
            MLPelements = config['MLPExtendTuneElement']

        layers = MLPelements['Layers']
        lrs = [float(item) for item in MLPelements['LearningRate']]
        max_epoch = MLPelements['MaxEpoch']
        drops = MLPelements['DropRate']
        patience = MLPelements['Patience']

        empty_df = pd.DataFrame([])
        hype_set_n = 0
        for lyr in layers:
            for lr in lrs:
                for drop in drops:
                    min_loss_step = []
                    min_loss_mean_per_fold = []
                    for fold in range(len(self.train_df)):
                        hypeinfo = f"Fold{fold}_Layers{lyr}_Learning_Rate{lr}_DropOuts_{drops}"
                        print(lyr, lr, drop, fold)
                        MLPtrainer = MLPTrainer(self.seed,
                                                lyr,
                                                lr,
                                                drop,
                                                max_epoch,
                                                fold,
                                                patience,
                                                self.device, 
                                                self.var_mode, 
                                                512,
                                                hypeinfo,
                                                early_stop=True)
                        min_loss, min_loss_epoch, _, mlp = MLPtrainer.fit(self.train_df[fold],self.valid_df[fold])
                        min_loss_step.append([hype_set_n,
                                              fold,
                                              lyr,
                                              lr,
                                              drop,
                                              min_loss,
                                              min_loss_epoch])
                        min_loss_mean_per_fold.append(min_loss)
                    this_fold_min_loss_mean = np.array(min_loss_mean_per_fold).mean()
                    hype_set_n += 1
                    perf_df = pd.DataFrame(min_loss_step,
                                           columns=['HypeSetNum',
                                                    'Fold',
                                                    'Layers',
                                                    'LearningRate',
                                                    'DropRates',
                                                    'MinTestLoss',
                                                    'MinEpochTestLoss'])
                    perf_df['MeanMinLoss'] = this_fold_min_loss_mean
                    empty_df = pd.concat([empty_df, perf_df]).reset_index(drop=True)
                    empty_df.to_csv(self.path_guide.mlp_tune_record)

        return empty_df
                    

class MLPTrainer:
    def __init__(self,
                 seed: int,
                 layer: list,
                 lr: float,
                 drop: list,
                 epoch: int,
                 fold: int,
                 patience: int,
                 device: str,
                 var_mode: str,
                 batch: int, # not available
                 hypeinfo : str,
                 early_stop: int = True
                 ):
        self.seed = seed
        self.layer = layer
        self.lr = lr
        self.drop = drop
        self.epoch = epoch
        self.fold = fold
        self.patience = patience
        self.device = device
        self.var_mode = var_mode
        self.scaler = StandardScaler()
        self.batch = batch
        self.hypeinfo = hypeinfo
        self.output_shape = 2
        self.early_stop = early_stop

    def train_one_epoch(self, mlp, tr_X, tr_y, opt, loss_fn):
        X = torch.tensor(tr_X).to(torch.float).to(self.device)
        if type(tr_y) == np.ndarray:
            y = torch.tensor(tr_y).long().to(self.device)
        elif type(tr_y) == torch.Tensor:
            y = tr_y.long().to(self.device)
        output = mlp(X)
        loss = loss_fn(output, y[:,1]) # if CRE
        loss.backward()
        opt.step()
        return loss.item()

    def early_stopping(self, cur_epoch, cur_val_loss, best_val_loss, best_epoch, p_counter):
        if cur_val_loss < best_val_loss: # update
            best_val_loss = cur_val_loss
            best_epoch = cur_epoch
            p_counter = 0
        else:
            p_counter += 1
        if p_counter > self.patience: #
            return 1, p_counter, best_val_loss, best_epoch
        return 0, p_counter, best_val_loss, best_epoch
            
    def eval(self, mlp, test, scaler):
        te_X, te_y = get_feature_and_target(test, 'test', 'mlp', scaler)
        metric = Metric('MLP')
        te_X = torch.tensor(te_X).to(torch.float).to(self.device)

        mlp.eval()
        mlp_output = mlp(te_X).detach().cpu().numpy()
        auroc, auprc = metric.get_auroc_and_auprc(mlp_output[:,1], 
                                                  te_y,
                                                  save_path = './results/figures/',
                                                  auc_figure=True,
                                                  auprc_figure=True)

        return auroc, auprc, mlp_output
    
    def get_scaler(self,train):
        tr_X, tr_y, self.scaler = get_feature_and_target(train, 'train', 'kan', self.scaler)
        return self.scaler
        
    def fit(self, train, valid):
        cls_weights = class_weight(train)
        tr_X, tr_y, self.scaler = get_feature_and_target(train, 'train', 'kan', self.scaler)
        vl_X, vl_y = get_feature_and_target(valid, 'valid', 'kan', self.scaler)
        
        cre_loss = nn.CrossEntropyLoss(weight=cls_weights.to(self.device))

        mlp = MLP(tr_X.shape, self.output_shape, self.drop, self.layer).to(self.device)
        cre_loss = nn.CrossEntropyLoss(weight=cls_weights.to(self.device))
        opt = torch.optim.Adam(mlp.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        history = []
        mlp.train()
        best_val_loss = float('inf')
        best_epoch = 0
        p_counter = 0

        progress_bar = tqdm(total=self.epoch, desc="Training")
        for epoch in range(self.epoch):
            cur_loss = self.train_one_epoch(mlp, tr_X, tr_y, opt, cre_loss)
            val_output = mlp(torch.tensor(vl_X).to(torch.float).to(self.device))
            val_loss = cre_loss(val_output, torch.tensor(vl_y[:,1]).long().to(self.device)).item()

            if self.early_stop:
                stop_sign, p_counter, best_val_loss, best_epoch = self.early_stopping(epoch, val_loss, best_val_loss, best_epoch, p_counter)
                if stop_sign == 1:
                    break

            progress_bar.set_description(f"Epoch {epoch+1}/{self.epoch}")
            progress_bar.set_postfix(train_loss=cur_loss, val_loss=val_loss)
            progress_bar.update(1)
        
        return val_loss, best_epoch, self.scaler, mlp











