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

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False

        
class Logistic(nn.Module):
    def __init__(self, in_dim, lr=0.001):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(in_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        
    def forward(self,x):
        return torch.sigmoid(self.linear(x)).squeeze()
    
    def fit(self, X, y, epochs=20000):
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        
        self.to('cuda:0')
        X = torch.tensor(X).to(dtype=torch.float).to('cuda:0')
        y = torch.tensor(y).to(dtype=torch.float).to('cuda:0')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
    def predict_proba(self,X):
        X = torch.tensor(X).to(dtype=torch.float).to('cuda:0')
        with torch.no_grad():
            y_prob = self.forward(X)
            return y_prob.detach().cpu().numpy()
        
        
        

    
        
class MLMTrainer: # Machine Learning Methods
    def __init__(self,
                 seed: int,
                 var_mode : str,
                 model_name : str,
                 input_dim: int,
                 path_guide):
        self.model_name = model_name
        self.var_mode = var_mode
        self.scaler = StandardScaler()
        '''self.lgr = LogisticRegression(solver='l',
                                      penalty='l1',
                                      max_iter=100,
                                      random_state=seed)'''
        self.lgr = LogisticRegression()
        self.lda = LDA()
        self.rfc = RandomForestClassifier(n_estimators=100, random_state=seed)
        
        self.lgr_cus = Logistic(input_dim,lr=0.001)
        set_seed(seed)
        
    def eval(self,test):
        te_X, te_y = get_feature_and_target(test, 'test', 'mlm', self.scaler)
        
        if self.model_name == 'lgr':
            fitted_model = self.lgr
            lgr_prob = self.lgr.predict_proba(te_X)
            lgr_metric = Metric('LogistiRegression')
            auc, apc = lgr_metric.get_auroc_and_auprc(lgr_prob[:,1], 
                                           te_y, 
                                           save_path = './results/figures/', 
                                           auc_figure=True,
                                           auprc_figure=True)
            return auc, apc, lgr_prob
        
        elif self.model_name == 'lda':
            fitted_model = self.lda
            lda_prob = self.lda.predict_proba(te_X)
            lda_metric = Metric('LinearDiscriminantAnalysis')
            auc, apc = lda_metric.get_auroc_and_auprc(lda_prob[:,1], 
                                                       te_y, 
                                                       save_path = './results/figures/', 
                                                       auc_figure=True,
                                                       auprc_figure=True)
            return auc, apc, lda_prob
        
        elif self.model_name == 'for':
            fitted_model = self.rfc
            rfc_prob = self.rfc.predict_proba(te_X)
            rfc_metric = Metric('RandomForest')
            auc, apc = rfc_metric.get_auroc_and_auprc(rfc_prob[:,1], 
                                                       te_y, 
                                                       save_path = './results/figures/', 
                                                       auc_figure=True,
                                                       auprc_figure=True)
            return auc, apc, rfc_prob
        elif self.model_name == 'lgr_custom':
            fitted_model = self.lgr_cus
            fitted_model.eval()
            lgr_prob = self.lgr_cus.predict_proba(te_X)
            lgr_metric = Metric('LogisticCustom')
            auc, apc=  lgr_metric.get_auroc_and_auprc(lgr_prob,
                                                      te_y,
                                                      save_path = './results/figures',
                                                      auc_figure=True,
                                                      auprc_figure=True)
            return auc, apc, lgr_prob
            
            
        else:
            print("ERROR")
        #, fitted_model
    
    def eval_logit_cus(self, test, scaler ,loaded_model):
        
        te_X, te_y = get_feature_and_target(test, 'test', 'mlm', scaler)

        fitted_model = loaded_model
        fitted_model.eval()
        lgr_prob = fitted_model.predict_proba(te_X)
        lgr_metric = Metric('LogisticCustom')
        auc, apc= lgr_metric.get_auroc_and_auprc(lgr_prob,
                                                  te_y,
                                                  save_path = './results/figures',
                                                  auc_figure=True,
                                                  auprc_figure=True)
        return auc, apc, lgr_prob

    
    def fit(self, train):
        tr_X, tr_y, self.scaler = get_feature_and_target(train, 'train', 'mlm', self.scaler)
          
        if self.model_name == 'lgr':      
            self.lgr.fit(tr_X, tr_y[:,1])
            return self.lgr
        elif self.model_name == 'lda':
            self.lda.fit(tr_X, tr_y[:,1])
            return self.lda
        elif self.model_name == 'for':
            self.rfc.fit(tr_X, tr_y[:,1])
            return self.rfc
        elif self.model_name == 'lgr_custom':
            self.lgr_cus.train()
            self.lgr_cus.fit(tr_X,tr_y[:,1])
            return self.lgr_cus
    
    
    
    
    
        