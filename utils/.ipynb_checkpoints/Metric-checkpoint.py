from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc


import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


class Metric:
    def __init__(self,
                 model_name: str,
                 ):
        self.model_name =  model_name

    def get_formula_output(self, kan, X):

        lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs', #
               'x^5', '1/x', '1/x^2','1/x^3','1/x^4','1/x^5','x^0.5','x^1.5','1/sqrt(x)']
        kan.auto_symbolic(lib=lib, verbose=0)
        formula1, formula2 = kan.symbolic_formula()[0]
        
        prob = []
        var_arr = []
        
        for sym in formula2.free_symbols:
            var_arr.append(str(sym))
        var_arr.sort(key=lambda x: int(x.split('_')[1]))
        var_symbol_str = ' '.join(var_arr)
        f = sp.lambdify(sp.symbols(var_symbol_str), formula2, "numpy")
        formula_output = f(*[X[:, i] for i in range(len(var_arr))])
        final_output = torch.sigmoid(torch.tensor(formula_output))
        return final_output, formula2

    def get_auroc_and_auprc(self, 
                            output: np.array or torch.tensor,
                            y : np.array,
                            save_path: str = None, 
                            auc_figure: bool =False,
                            auprc_figure: bool=False):
        if y.ndim != 2:
            raise ValueError("This function only accepts 2D arrays.")

        fpr, tpr, thresholds =  roc_curve(y[:,1], output)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y[:,1], output)
        auprc = auc(recall, precision)
        
        
        if auc_figure:
            plt.figure()
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"KAN AUROC = {roc_auc:.4f}")
            plt.title(f'ROC Curve (AUROC = {roc_auc:.4f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(save_path+self.model_name+'_AUROC_from_RandomTestSet.png')
        
        if auprc_figure:
            plt.figure()
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, label=f"KAN AUPRC = {auprc:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(save_path+self.model_name+'_AUPRC_from_RandomTestSet.png')
        
        
        return roc_auc, auprc


    def foo(self):
        pass





