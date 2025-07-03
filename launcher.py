
from new_kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import pandas as pd
from utils.Loader_for_mlp import Loader
from utils.KFolds_splitting import KfoldSplitter, split_train_test
from trainer.KANTurainer import *
from trainer.MLPTurainer import MLPTuner, MLPTrainer
from trainer.MLMTrainer import MLMTrainer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import os
import argparse
from datetime import datetime
import pickle
import yaml



class GUIDE():
    def __init__(self):
        self.csv_path = './data/csv/NEW_ASCVD_2M_unique_gfr_selection_with_year_BFC_myvar_after_gfr_imputation.csv'
        self.var_path = './cfg/RandomVariables.yaml'
        self.cfg_path = './cfg/TuningElement.yaml'
        self.seed = 0
        self.fold = 0
        
    def dynamic_path(self, var_mode ,formatted_now):
        self.formatted_now = formatted_now
        self.var_mode = var_mode
        self.mlm_logcus = f'./results/weights/Logit_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_trained_{self.formatted_now}.pth'
        self.mlp_weight = f'./results/weights/MLP_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_trained_{self.formatted_now}.pth'
        self.mlp_tune = f'./results/tuning/MLP_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_hypetune_results_{self.formatted_now}.csv'
        self.mlp_tune_record = f'./results/tuning/MLP_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_hypetune_For_Record.csv'
        self.mlp_weight_path = f'./results/weights/MLP_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_weight_{self.formatted_now}.pth'
        self.kan_weight = f'./results/weights/KAN_Seed_{self.seed}_{self.var_mode}_trained_formula_{self.formatted_now}.txt'
        self.kan_tune = f'./results/tuning/KAN_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_hypetune_results_{self.formatted_now}.csv'
        self.kan_tune_record = f'./results/tuning/KAN_Seed_{self.seed}_Fold_{self.fold}_{self.var_mode}_hypetune_For_Record.csv'
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 설정
        torch.backends.cudnn.deterministic = True  # CUDNN 결정론적 동작 설정
        torch.backends.cudnn.benchmark = False

def load_data(seed, var_mode, csv_path, var_path):
    with open(var_path, 'r') as file:
        config = yaml.safe_load(file)
    loader = Loader(config[var_mode], csv_path)
    df = loader.csv_load
    
    filtered_df = loader.remove_outliers(df)
    filtered_train_df, filtered_test_df = train_test_split(filtered_df, test_size=0.2, random_state=seed)
    
    KFOLDS = loader.make_KFoldData(filtered_train_df, balanced_test=False)
    train, valid = KFOLDS[0],KFOLDS[1]
    return train, valid, filtered_test_df

def data_info_print(test, train, valid):

    data_info = ""
    for fold in range(len(train)):
        tr_df = train[fold]
        vl_df = valid[fold]
        data_info+= f"##########################{fold}-th fold Data Info ##########################\n"
        data_info += f"Test info{test.shape} -> ASCVD Ratio:,{round(test[test['ASCVD']==1].shape[0] /test[test['ASCVD']==0].shape[0],5)} ASCVD N: {test[test['ASCVD']==1].shape[0]} Non-ASCVD N: {test[test['ASCVD']==0].shape[0]}\n"
        data_info += f"Train info{tr_df.shape} -> ASCVD Ratio: {round(tr_df[tr_df['ASCVD']==1].shape[0] / tr_df[tr_df['ASCVD']==0].shape[0],5)} ASCVD N: {tr_df[tr_df['ASCVD']==1].shape[0]} Non-ASCVD N: {tr_df[tr_df['ASCVD']==0].shape[0]}\n"
        data_info += f"Valid info{vl_df.shape} -> ASCVD Ratio: {round(vl_df[vl_df['ASCVD']==1].shape[0] / vl_df[vl_df['ASCVD']==0].shape[0],5)} ASCVD N: {vl_df[vl_df['ASCVD']==1].shape[0]} Non-ASCVD N: {vl_df[vl_df['ASCVD']==0].shape[0]}\n"
    data_info += "########################################################################"
    print(data_info)


def kan_launcher(train, valid, test, seed, var_mode, kan_opt, device, path_guide):
    epoch_select_method = 'max' # min, mean, max

    kan_tuner = KANTuner(train, 
                         valid, 
                         seed, 
                         var_mode, 
                         kan_opt, 
                         device, 
                         2048, 
                         path_guide)
    
    search_results = kan_tuner.hype_search()
    search_results.to_csv(path_guide.kan_tune)
    min_index = search_results['MeanMinLoss'].idxmin()
    min_row = search_results.loc[min_index]
    best_hype_set = search_results[search_results['MeanMinLoss']==min_row['MeanMinLoss']]

    print(best_hype_set['MinEpochTestLoss'])
    if epoch_select_method == 'max':
        best_max_epoch = best_hype_set['MinEpochTestLoss'].max()
    elif epoch_select_method == 'mean':
        best_max_epoch = int(best_hype_set['MinEpochTestLoss'].mean())
    elif epoch_select_method == 'min':
        best_max_epoch = best_hype_set['MinEpochTestLoss'].min()
    else:
        raise ValueError("Wrong value.")

    print(best_max_epoch)
    KANtrainer = KANTrainer(seed,
                            min_row['Layers'],
                            min_row['LearningRate'], 
                            best_max_epoch,
                            min_row['Order'],
                            min_row['Grid'],
                            min_row['Lambda'],
                            min_row['LambdaEntropy'], 
                            -1,
                            kan_opt,
                            device, 
                            var_mode, 
                            2048, 
                            'FINAL_KAN',
                            True, # Model Figure Save
                           )
    fit_results, kan, scaler = KANtrainer.fit(pd.concat([train[0],valid[0]]).reset_index(drop=True), # train + valid = whole training set
                                              test)
    auroc, auprc, kan_formula, kan_output = KANtrainer.eval(kan, test, scaler)
    result_write = f"AUC: {auroc}\nAUPRC: {auprc}\nFormula: {kan_formula}"
    with open(path_guide.kan_weight, "w") as txt_file:
        txt_file.write(result_write)
    
    
    return auroc, auprc


def mlp_launcher(train, valid, test, seed, var_mode, device, path_guide):
    mlp_tuner = MLPTuner(train, valid, seed, var_mode, device, 1, path_guide)
    search_results = mlp_tuner.hype_search()
    search_results.to_csv(path_guide.mlp_tune)
    min_index = search_results['MeanMinLoss'].idxmin()
    min_row = search_results.loc[min_index]
    best_hype_set = search_results[search_results['MeanMinLoss']==min_row['MeanMinLoss']]
    best_max_epoch = best_hype_set['MinEpochTestLoss'].max()

    MLPtrainer = MLPTrainer(seed,
                            min_row['Layers'],
                            min_row['LearningRate'],
                            min_row['DropRates'],
                            best_max_epoch,
                            -1, # fold
                            0, # patience
                            device, 
                            var_mode, 
                            512, # batch -> not available
                            'FINAL_MLP',
                            early_stop = False)
    
    final_train = pd.concat([train[0],valid[0]]).reset_index(drop=True)
    print(final_train.shape, test.shape)
    min_loss, min_loss_epoch, scaler, mlp = MLPtrainer.fit(final_train, test)
    torch.save(mlp, path_guide.mlp_weight_path) 
    loaded_mlp = torch.load(path_guide.mlp_weight_path)
    
    auroc, auprc, mlp_output = MLPtrainer.eval(loaded_mlp, test, scaler)
    return auroc, auprc


def mlm_launcher(train, valid, test, seed, var_mode, model ,path_guide,):
    
    mlm_trainer = MLMTrainer(seed,
                             var_mode,
                             model,
                             train[0].shape[1] - 3, # - ascvd, etc...s
                             path_guide)
    mlm_model = mlm_trainer.fit(pd.concat([train[0],valid[0]]).reset_index(drop=True))
    if model == 'lgr_custom':
        torch.save(mlm_model, path_guide.mlm_logcus) 
        loaded_lgr = torch.load(path_guide.mlm_logcus)
        auroc, auprc, mlm_prob = mlm_trainer.eval(test)
        print("Its Logistic Regression Test Performance:", auroc, auprc)
        
    mlm_auc, mlm_apc, mlm_output = mlm_trainer.eval(test)
    return mlm_auc, mlm_apc

def main(args):
    seed = args.seed
    model_name = args.model
    verbose = args.verbose
    kan_opt = args.opt
    ablation = args.ablation
    var_mode= args.var_type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    set_seed(seed)
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d_%H%M%S")
    
    data_info = ""
    path_guide = GUIDE()

    train, valid, test = load_data(seed, var_mode, path_guide.csv_path, path_guide.var_path)
    path_guide.dynamic_path(var_mode, formatted_now)

    print(test)
    if verbose:
        data_info_print(test, train, valid)

    if ablation:
        print("Do ablation study")
    
    if model_name == "mlp" or model_name == "MLP":
        mlp_auc, mlp_apc = mlp_launcher(train, valid, test, seed, var_mode, device, path_guide)
        mlp_print = f"\nMLP AUC:{mlp_auc}, APC:{mlp_apc}"
        print(mlp_print)
    elif model_name == "kan" or model_name == "KAN":
        kan_auc, kan_apc = kan_launcher(train, valid, test, seed, var_mode, kan_opt ,device, path_guide)
        kan_print = f"\nKAN AUC:{kan_auc}, APC:{kan_apc}"
        print(kan_print)
    elif model_name == "mlm" or model_name == "MLM":
        lgr_cus_auc, lgr_cus_apc = mlm_launcher(train, valid, test, seed, var_mode, 'lgr_custom' ,path_guide)
        #lgr_auc, lgr_apc = mlm_launcher(train, valid, test, seed, var_mode, 'lgr', path_guide)
        lda_auc, lda_apc = mlm_launcher(train, valid, test, seed, var_mode, 'lda', path_guide)
        for_auc, for_apc = mlm_launcher(train, valid, test, seed, var_mode, 'for', path_guide)
        lgr_print = f"Logit AUC:{lgr_cus_auc}, APC:{lgr_cus_apc}"
        lda_print = f"LDA AUC:{lda_auc}, APC{lda_apc}"
        for_print = f"Forest AUC:{for_auc}, APC:{for_apc}"
    else:
        raise ValueErorr("Invalid model name")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--seed', type=int, default=0 ,help='Input Random Seed')
    parser.add_argument('--model', type=str, choices=["kan", "mlp", "mlm"], required=True,help='mlp or kan or mlm')
    parser.add_argument('--verbose',type=int, default=0, help='Input wheter the verbose or not')
    parser.add_argument('--opt', type=str, default='Adam', choices = ["Adam","LBGFS"], help="select LBFGS or Adam")
    parser.add_argument('--ablation', type=int, default=0, help="If you want only ablation study use trained model, input 1")
    parser.add_argument('--var_type', type=str, default='EXTEND', help="Input wheter the PRIMARY or EXTEND")
    args = parser.parse_args()
    main(args)




