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
from new_kan import *
from sklearn.impute import KNNImputer
import torch.optim as optim
from utils.Metric import *
from utils.Loader_for_mlp import *

print("Import KANTurainer")

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


class KANTuner:
    def __init__(self,
                 train_df : pd.DataFrame,
                 valid_df : pd.DataFrame,
                 seed : int,
                 var_mode: str,
                 opt : str,
                 device: str,
                 batch: int,
                 path_guide
                 ):
        self.train_df = train_df
        self.valid_df = valid_df
        self.var_mode = var_mode
        self.seed = seed
        self.opt = opt
        self.device = device
        self.scaler = StandardScaler()
        set_seed(self.seed)
        self.batch = batch 
        self.path_guide = path_guide

    def hype_search(self):
        with open(self.path_guide.cfg_path, 'r') as file:
            config = yaml.safe_load(file)
        if self.var_mode == 'PRIMARY':
            KANelements = config['KANPrimaryTuneElement']
        elif self.var_mode == 'EXTEND':
            KANelements = config['KANExtendTuneElement']

        layers = KANelements['Layers']
        lrs = [float(item) for item in KANelements['LearningRate']]
        grids = KANelements['Grids']
        orders = KANelements['Orders']
        max_step = KANelements['MaxStep']
        lambdas = KANelements['Lambdas']
        lambdas_entropy = KANelements['LambdaEntropy']

        empty_df = pd.DataFrame([])
        hype_set_n = 0
        for grid in grids:
            for order in orders:
                for lde in lambdas_entropy:
                    for ld in lambdas:
                        for lyr in layers:
                            for lr in lrs:
                                # start iteration fold of hyperparam
                                
                                print(type(ld), type(lde), type(max_step))
                                
                                min_loss_step = []
                                min_loss_mean_per_fold = []
                                for fold in range(len(self.train_df)):
                                    self.path_guide.fold = fold
                                    hypeinfo = f"Fold{fold}_Grid{grid}_Order{order}_LambdaEntropy{lde}_Lambdas:{ld}_Layers{lyr}_Learning_Rate{lr}_{self.opt}"
                                    print(hypeinfo)
                                    KANtrainer = KANTrainer(self.seed,
                                                            lyr,
                                                            lr, 
                                                            max_step, 
                                                            order, 
                                                            grid, 
                                                            ld,
                                                            lde, 
                                                            fold,
                                                            self.opt,
                                                            self.device, self.var_mode, self.batch, hypeinfo)
                                    fit_results, kan, scaler = KANtrainer.fit(self.train_df[fold],
                                                                              self.valid_df[fold])

                                    loss_plot_path = f"./results/Loss_{hypeinfo}.png"

                                    
                                    print("Minimum Loss step:", np.array(fit_results['test_loss']).argmin(),
                                          "Minimum Loss:", np.array(fit_results['test_loss']).min())
                                    print("Mean Loss:", np.array(fit_results['test_loss']).mean())
                                    
                                    min_loss_step.append([hype_set_n,
                                                          fold,
                                                          grid,
                                                          order,
                                                          lde,
                                                          ld,
                                                          lyr,
                                                          lr,
                                                          np.array(fit_results['test_loss']).argmin(),
                                                          np.array(fit_results['test_loss']).min(),
                                                          np.array(fit_results['test_loss']).mean()])
                                    min_loss_mean_per_fold.append(np.array(fit_results['test_loss']).min())
                                hype_set_n+=1
                                this_fold_min_loss_mean = np.array(min_loss_mean_per_fold).mean()
                                perf_df = pd.DataFrame(min_loss_step,
                                                       columns = [
                                                                'HypeSetNum',
                                                                'Fold',
                                                                'Grid',
                                                                'Order',
                                                                'LambdaEntropy',
                                                                'Lambda',
                                                                'Layers',
                                                                'LearningRate',
                                                                'MinEpochTestLoss',
                                                                'MinTestLoss',
                                                                'MeanTestLoss'])
                                perf_df['MeanMinLoss'] = this_fold_min_loss_mean
                                min_loss_mean = perf_df['MinTestLoss'].mean()
                                empty_df = pd.concat([empty_df, perf_df]).reset_index(drop=True)
                                empty_df.to_csv(self.path_guide.kan_tune_record)
                            
        return empty_df
                                


class KANTrainer:
    def __init__(self,
                 seed: int,
                 layer: list,
                 lr: float,
                 step: int,
                 order: int,
                 grid: int,
                 lamb: float,
                 lamb_entropy: float,
                 fold: int,
                 opt: str,
                 device: str,
                 var_mode: str,
                 batch: int,
                 hypeinfo : str,
                 model_fig : bool = False
                 ):
        self.seed = seed
        self.layer = layer
        self.lr = lr
        self.step = step
        self.order = order
        self.grid = grid
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.fold = fold
        self.device = device
        self.var_mode = var_mode
        self.scaler = StandardScaler()
        self.batch = batch
        self.hypeinfo = hypeinfo
        self.opt = opt
        self.model_fig = model_fig

    def get_kan_dataset(self, tr_X, tr_y, vl_X, vl_y):
        dataset = {}
        dtype = torch.get_default_dtype()
        dataset['train_input'] = torch.from_numpy(tr_X).type(dtype).to(self.device)
        dataset['test_input'] = torch.from_numpy(vl_X).type(dtype).to(self.device)
        dataset['train_label'] = torch.from_numpy(tr_y[:,1]).type(torch.long).to(self.device)
        dataset['test_label'] = torch.from_numpy(vl_y[:,1]).type(torch.long).to(self.device)
        return dataset

    def eval(self, kan, test, scaler):
        te_X, te_y = get_feature_and_target(test, 'test', 'kan', scaler)
        metric = Metric('KAN')
        kan.eval()
        kan_output, kan_formula  = metric.get_formula_output(kan, te_X) # with sigmoid

        auroc, auprc = metric.get_auroc_and_auprc(kan_output, 
                                                  te_y, 
                                                  save_path = './results/figures/',
                                                  auc_figure=True,
                                                  auprc_figure=True)

        return auroc, auprc, kan_formula, kan_output
    
    def fit(self, train, valid):
        cls_weights = class_weight(train)
        tr_X, tr_y, self.scaler = get_feature_and_target(train, 'train', 'kan', self.scaler)
        vl_X, vl_y = get_feature_and_target(valid, 'valid', 'kan', self.scaler)

        dataset = self.get_kan_dataset(tr_X, tr_y, vl_X, vl_y)
        cre_loss = nn.CrossEntropyLoss(weight=cls_weights.to(self.device))
        
        kan = KAN(width = self.layer,
                  grid = self.grid, 
                  k = self.order, 
                  seed = self.seed,
                  device=self.device)

        
        if self.model_fig:
            kan(dataset['train_input'])
            plt.figure() 
            kan.plot(title="before_"+self.hypeinfo, save_graph=True)
        
        kan.train()
        print(f"LearningRate:{self.lr}, MaxSteps:{self.step}, Lambda:{self.lamb}, LambdaEntropy:{self.lamb_entropy}, Layer:{self.layer}, Grid:{self.grid}, Order(k):{self.order}")
        results = kan.fit(dataset,
                          opt=self.opt,
                          lr=self.lr,
                          steps=self.step,
                          #metrics=(train_auc, test_auc, test_balance_loss),
                          loss_fn = cre_loss,
                          lamb=self.lamb,
                          lamb_entropy=self.lamb_entropy,
                          update_grid=False,
                          batch=self.batch)
        kan = kan.prune()
        kan = kan.prune_node(threshold=1e-3)
        kan.prune_edge()
        if self.model_fig:
            plt.figure()
            kan.plot(title="after_"+self.hypeinfo, save_graph=True)
            plt.savefig('AFTER_KAN.png',dpi=600)
        return results, kan, self.scaler
        
                 



















