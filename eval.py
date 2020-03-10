import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
import tensorflow.keras.backend as K
np.random.seed(2020)
import hyper_params as hp
from dl_models import build_model_predict
from load_data import load_data
from load_data import impute_data, calc_impute_values

import shap

if __name__ == '__main__':
    # load dataset
    hp = hp.create_hparams()
    op_mode = hp.op_mode
    print(hp.outcome)
    pdirname = os.path.dirname(__file__)

    idx = hp.xrsval_model_to_use -1  #Model to use, based on cross-validation results (during training)

    clin_params, outcomes, patients_id = load_data(hp.outcome, pdirname + hp.dataset_path) # Load all datasets
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30,random_state=0)
    sss2 = StratifiedShuffleSplit(n_splits=10, test_size=0.28,random_state=0)
    print('Positives = {:.1f}'.format(np.sum(outcomes))+'   Negatives = {:.1f}'.format(np.sum(1-outcomes))+' Total = {:.1f}'.format(len(outcomes)))

    orig_clin_params = clin_params
    kk = sss1.split(clin_params, outcomes)  # only one split is available, dummy for-loop
    kk = list(kk)
    train_valid_index, test_index = kk[0] # we have only one data split of test vs train/validation
    trv_params = orig_clin_params[train_valid_index]  # 70% of data
    trv_outcomes = outcomes[train_valid_index]

    dummy =0
    for train_index, valid_index in sss2.split(trv_params, trv_outcomes):
        if dummy == idx:
            break # get only the first 'idx' splits
        else:
            dummy+=1

    impute_values = calc_impute_values(trv_params[train_index])  # calculate impute values from TRAIN data only
    clin_params = impute_data(X=orig_clin_params,impute_values=impute_values)  # impute the entire dataset based on training
    trv_params = clin_params[train_valid_index]

    K.clear_session() # just to ensure all graphical models are clear
    fn_prd = pdirname + hp.weights_path + '/model_' + hp.outcome + '_' + str(idx+1) + '.hdf5' # idx starts from 0; filenames start from 1
    model_prd = build_model_predict(clin_params.shape[1], hp.model_layers_ae, hp.model_layers_prd)  # Build a model
    model_prd.load_weights(fn_prd)# load weights of model performed best on the validation dataset (specified by idx)

    gt_outcomes = outcomes[test_index]
    prd_outcomes = model_prd.predict(clin_params[test_index])
    explainer_shap = shap.GradientExplainer(model_prd, trv_params)
    shap_values = explainer_shap.shap_values(clin_params[test_index])
    shap.summary_plot(shap_values[0],clin_params[test_index],plot_type='dot')

    # Display
    auc_ = []
    aucs = []
    tps  = []
    mean_fp = np.linspace(0, 1, 100)
    plt.figure(2, figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    fp, tp, thresholds = roc_curve(gt_outcomes, prd_outcomes)
    auc_ =auc(fp, tp)
    print('Area Under Curve = {:.3f}'.format(auc_))
    plt.plot(fp, tp, color='b', lw=2, alpha=0.8)
    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.01])
    plt.xlabel('')
    plt.ylabel('')
    a = hp.dataset_path
    k=[pos for pos, char in enumerate(a) if char == "/"]
    plt.title(a[k[-1]+12:-4]+'Param={:.1f} '.format(clin_params.shape[1])+' Pos={:.1f}'.format(np.sum(outcomes))+' Neg={:.1f}'.format(np.sum(1-outcomes)), fontsize=8)
    plt.legend(loc="lower right",prop={'size': 6} )
    plt.grid()
    plt.savefig(pdirname+'/results/'+a[k[-1]+1:-4]+".png", dpi=300)
    plt.show()