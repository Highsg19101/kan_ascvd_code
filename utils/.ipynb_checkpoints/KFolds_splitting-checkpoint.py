import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import copy
import random


def split_train_test(df : pd.DataFrame, ratio: float):
    pos = df[df['ASCVD']==1.0]
    neg = df[df['ASCVD']==0.0]
    
    boundary = int(pos.shape[0]*ratio) # consider imbalance character
    
    test_pos = df[df['ASCVD']==1.0][:boundary]
    test_neg = df[df['ASCVD']==0][:boundary]
    
    train_pos = df[df['ASCVD']==1.0][boundary:]
    train_neg = df[df['ASCVD']==0][boundary:]

    train = pd.concat([train_pos,train_neg])
    test = pd.concat([test_pos, test_neg])
    
    return train.sample(frac=1).reset_index(drop=True), test.sample(frac=1).reset_index(drop=True)

def oversampling(df):
    pos = df[df['ASCVD']==1.0]
    neg = df[df['ASCVD']==0.0]
    multiple_num = int(neg.shape[0]/pos.shape[0])
    oversampling_pos =  pd.concat([pos] * multiple_num, ignore_index=True).reset_index(drop=True)
    oversampling_after = pd.concat([oversampling_pos, neg],ignore_index=True).sample(frac=1).reset_index(drop=True)
    return oversampling_after
    

class KfoldSplitter:
    def __init__(self, Knum: int, random_seed : int):
        self.knum = Knum
        self.seed = random_seed
       
    def fivefold_split_unbalanced(self, hme_df: pd.DataFrame):
        hme_df = hme_df.sample(frac=1).reset_index(drop=True)
        kf = KFold(n_splits=self.knum, shuffle=True, random_state=self.seed)
        
        tr_five_folds, vl_five_folds = [],[]
        for fold, (train_index, val_index) in enumerate(kf.split(hme_df)):
            train_df = hme_df.iloc[train_index]
            val_df = hme_df.iloc[val_index]
            tr_five_folds.append(train_df)
            vl_five_folds.append(val_df)
        return tr_five_folds, vl_five_folds
    
    def fivefold_split(self, hme_df: pd.DataFrame):
        hme_df = hme_df.sample(frac=1).reset_index(drop=True)
        
        # imbalance issue
        
        pos = hme_df[hme_df['ASCVD']==1]
        neg = hme_df[hme_df['ASCVD']==0]
    
        pos_tr_five_folds = []
        pos_vl_five_folds = []
        
        neg_tr_five_folds = []
        neg_vl_five_folds = []

        # positive에서 4:1 * 5 set 만들고
        kf = KFold(n_splits=self.knum, shuffle=True, random_state=self.seed)
        for fold, (train_index, val_index) in enumerate(kf.split(pos)):
            train_df = pos.iloc[train_index]
            val_df = pos.iloc[val_index]
            
            pos_tr_five_folds.append(train_df)
            pos_vl_five_folds.append(val_df)

        pivot = pos_vl_five_folds[0].shape[0]
        for fold, (train_index, val_index) in enumerate(kf.split(neg)):
            train_df = neg.iloc[train_index]
            val_df = neg.iloc[val_index]
    
            train_df = pd.concat([train_df, val_df[pivot:]]) #
            val_df = val_df[:pivot]
            
            neg_tr_five_folds.append(train_df)
            neg_vl_five_folds.append(val_df)
        
        print(pos_tr_five_folds[0].shape[0],
              neg_tr_five_folds[0].shape[0],
              pos_vl_five_folds[0].shape[0],
              neg_vl_five_folds[0].shape[0])
        
        return pos_tr_five_folds, neg_tr_five_folds, pos_vl_five_folds, neg_vl_five_folds

    def merge_pos_neg(self, pos_fold : list, neg_fold: list):
        arr = []
        positive_num =  pos_fold[0].shape[0]
        negative_num =  neg_fold[0].shape[0]
        over_ratio = int(negative_num/positive_num)
        print("Pos Shape",positive_num, "Neg Shape",negative_num, "Ratio:",int(negative_num/positive_num))

        for i in range(len(pos_fold)):
            new_df = pd.concat([pos_fold[i], neg_fold[i]])
            arr.append(new_df)
        return arr
        
    
    def split_validation(self, tr_list: list, vl_list: list):
        for i in range(self.knum):
            common = pd.merge(tr_list[i], 
                              vl_list[i], on='INDI_DSCM_NO')
            if common.shape[0] == 0:
                pass
                
            else:
                raise ValueError("Error: There are overlapping data between the training set and the validation set.")











        
        