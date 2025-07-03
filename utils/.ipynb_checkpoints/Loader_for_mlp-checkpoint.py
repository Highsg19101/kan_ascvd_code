import pandas as pd
import torch
import warnings
import numpy as np
from utils.KFolds_splitting import *
from sklearn.impute import KNNImputer


warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_bin_label(input_list: np.ndarray) -> torch.tensor:
    binary_labels = [[1, 0] if x == 0 else [0, 1] for x in input_list]
    return torch.from_numpy(np.array(binary_labels)).to(torch.float)

def KANget_bin_label(input_list: np.ndarray) -> torch.tensor:
    binary_labels = [[1, 0] if x == 0 else [0, 1] for x in input_list]
    return np.array(binary_labels)


def get_feature_and_target(df: pd.DataFrame,
                       forwhat: str, 
                       model_name: str,
                       scaler):
    if forwhat == 'train':
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
    else:
        df_shuffled = df
    
    X = df_shuffled.drop(['INDI_DSCM_NO','HME_DT','ASCVD'],axis=1).values
    y = df_shuffled[['ASCVD']].values.flatten().astype(int)
    
    if forwhat == 'train':
        scaler.fit(X) # train이면 fit 추가
        X = scaler.transform(X)
        if model_name == 'mlp' or model_name == 'mlm':
            return X, get_bin_label(y), scaler
        elif model_name == 'kan':
            return X, KANget_bin_label(y), scaler
        else:
            raise ValueError(f"Input model name must be either 'mlp' or 'kan' or 'mlm', but got {te_y.shape}")

        
    elif forwhat == 'valid' or forwhat == 'test':
        X = scaler.transform(X)
        if model_name == 'mlp' or model_name == 'mlm':
            return X, get_bin_label(y)
        elif model_name == 'kan':
            return X, KANget_bin_label(y)
        else:
            raise ValueError(f"Input model name must be either 'mlp' or 'kan' or 'mlm', but got {te_y.shape}")

    else:
        raise ValueError("Please input to [train] or [valid] or [test].")


class Loader:
    def __init__(self, 
                 variables: list,
                 rdata_path : str):
        self.variables = variables
        self.rdata_path = rdata_path
            
    def make_KFoldData(self, df, balanced_test = True ,fold_seed=0):
        splitter = KfoldSplitter(5, fold_seed)
        
        if balanced_test:
            pos_tr_list,  neg_tr_list, pos_vl_list, neg_vl_list = splitter.fivefold_split(df)
            splitter.split_validation(pos_tr_list, pos_vl_list)
            splitter.split_validation(neg_tr_list, neg_vl_list)

            train_arr = splitter.merge_pos_neg(pos_tr_list, neg_tr_list)
            test_arr = splitter.merge_pos_neg(pos_vl_list, neg_vl_list)


        else:
            train_arr, test_arr = splitter.fivefold_split_unbalanced(df)
            
        return [train_arr, test_arr]

    @property
    def csv_load(self):
        result = pd.read_csv(self.rdata_path)
        return result[self.variables]

    def remove_outliers(self,df):
        ranges = {
        'AGE': (30, 79),
        'SEX_TYPE_org': (0, 1),
        'G1E_BMI': (5, 70),
        'G1E_BP_SYS': (70, 250),
        'G1E_HDL': (20, 100),
        'G1E_TOT_CHOL': (100, 400),
        'SMK': (0, 1),
        'DM': (0, 1),
        'HTN_med': (0, 1),
        'G1E_BP_DIA': (40, 150),
        'G1E_URN_PROT': (0, 6),
        'G1E_TG': (30, 1000),
        'G1E_HGB': (6, 20),
        'G1E_SGOT': (0, 1000),
        'G1E_SGPT': (0, 1000),
        'G1E_GGT': (0, 1000),
        'G1E_FBS': (30, 500),
        'G1E_CRTN': (0.1, 15),
        'G1E_LDL': (30, 300),
        'G1E_GFR': (0,250),
        'G1E_WSTC': (30, 250),
        'CALC_CTRB_VTILE_FD':(0,20),
        'RVSN_ADDR_CD':(11110.0, 50130.0)
        }
        for column, (min_val, max_val) in ranges.items():
            if column in df.columns:
                if column == 'G1E_GFR':
                    print(column,':',df.shape[0],'---->',end='')
                    pre_n = df.shape[0]
                    pre_pos = df[df['ASCVD']==1].shape[0]
                    df = df[df[column].isna() | (df[column] >= min_val) & (df[column] <= max_val)]
                    print(df.shape[0], end='')
                    post_n = df.shape[0]
                    post_pos = df[df['ASCVD']==1].shape[0]
                    print(' : Outliar Filtering:',pre_n - post_n, ',  ASCVD loss:',pre_pos-post_pos)
                    print("GFR None is", df['G1E_GFR'].isnull().sum())
                    gfr_ascvd = df[df['G1E_GFR'].isna()]
                    print("ASCVD in after filtering GFR None:",gfr_ascvd[gfr_ascvd['ASCVD']==1].shape[0])
                    continue
                print(column,':',df.shape[0],'---->',end='')
                pre_n = df.shape[0]
                pre_pos = df[df['ASCVD']==1].shape[0]

                df = df[(df[column] >= min_val) & (df[column] <= max_val)]
                print(df.shape[0], end='')
                post_n = df.shape[0]
                post_pos = df[df['ASCVD']==1].shape[0]
                print(' : Outliar Filtering:',pre_n - post_n, ',  ASCVD loss:',pre_pos-post_pos)
        return df
