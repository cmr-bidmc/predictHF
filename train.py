import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc
import tensorflow.keras.backend as K
np.random.seed(2020)
import hyper_params as hp
from dl_models import build_model_predict
from dl_models import build_model_ae
from load_data import load_data, impute_data, calc_impute_values

if __name__ == '__main__':
    # load dataset
    hp = hp.create_hparams()
    op_mode = hp.op_mode

    print(hp.outcome)
    pdirname = os.path.dirname(__file__)
    clin_params, outcomes, patients_id = load_data(hp.outcome, pdirname + hp.dataset_path) # Load dataset
    orig_clin_params = clin_params  # Imputation will change clin_params; so, we need to store it a copy separate

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=0)  #prepare cross-validation: training/validation (70%) + test (30%)
    sss2 = StratifiedShuffleSplit(n_splits=10, test_size=0.28, random_state=0) #=0.28x0.7 (20%) Validation + (50%) Training
    print('Positives = {:.1f}'.format(np.sum(outcomes))+'   Negatives = {:.1f}'.format(np.sum(1-outcomes))+' Total = {:.1f}'.format(len(outcomes)))

    ###################################################################################################
    ## Stage 1: PRE-TRAINING of AutoEncoder
    ###################################################################################################
    if op_mode is 'pretrain':
        idx =0
        for train_valid_index, test_index in sss1.split(clin_params, outcomes):#Dummy for-loop: only one split (test vs train/vlid) is performed.
            trv_params = orig_clin_params[train_valid_index]  # 70% of data; the other 30% are kept separate; not used for train or valid at all
            trv_outcomes = outcomes[train_valid_index]        # 70% of data
            for train_index, valid_index in sss2.split(trv_params, trv_outcomes): # for-loop on the 10-fold cross-validations
                trv_params = orig_clin_params[train_valid_index]  # 70% of data
                idx += 1
                fn_ae = pdirname + hp.weights_path + '/pretrained_model_' + hp.outcome + '_' + str(idx) + '.hdf5'
                # send the clinical params for imputation; include indices of train_index with them
                impute_values = calc_impute_values(trv_params[train_index])  # calculate impute values from TRAIN data only
                callbacks_ae = [
                    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001 , verbose=1),
                    ModelCheckpoint(filepath=fn_ae, verbose=1, save_best_only=True, save_weights_only=True) ]
                clin_params = impute_data(X=orig_clin_params,impute_values=impute_values)  # impute the entire dataset based on training dataset
                trv_params  = clin_params[train_valid_index]  # update after imputation

                K.clear_session() # to make sure that no data leaks among different cross-validations
                model_ae = build_model_ae(trv_params.shape[1],hp.model_layers_ae)  # Build an autorncoder model --> Feature Extraction
                history_ae = model_ae.fit(trv_params[train_index], trv_params[train_index],
                                          batch_size=hp.batch_size, epochs=hp.nr_epochs_ae,
                                          verbose=1,validation_data=(trv_params[valid_index],trv_params[valid_index]),
                                          shuffle=True, callbacks=callbacks_ae)
                plt.figure(2)
                plt.plot(history_ae.history['loss'])
                plt.plot(history_ae.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'valid'], loc='upper left')

        plt.show()
        sys.exit() # just do AE pre-train, save, and exit
    ###################################################################################################
    ###################################################################################################

    # Tou are here only if op_mode is NOT 'pretrain; i.e. train
    ###################################################################################################
    # Stage 2: training of Encoder-Predictor
    ###################################################################################################
    idx  = 0
    auc_ = []
    aucs = []
    tps  = []
    mean_fp = np.linspace(0, 1, 100)
    plt.figure(2, figsize=(4, 4))
    plt.plot([0, 1], [0, 1], 'k--')

    # Work on the same exact data-split as in autoEncoder part
    for train_valid_index, test_index in sss1.split(clin_params, outcomes):
        trv_params   = orig_clin_params[train_valid_index]
        trv_outcomes = outcomes[train_valid_index]
        for train_index, valid_index in sss2.split(trv_params, trv_outcomes):
            trv_params = orig_clin_params[train_valid_index]
            idx += 1
            impute_values = calc_impute_values(trv_params[train_index])  # calculate impute values from TRAIN data only
            clin_params = impute_data(X=orig_clin_params,impute_values=impute_values)  # impute the entire dataset based on training
            trv_params = clin_params[train_valid_index]  # update trv_params after imputation

            K.clear_session()# to make sure that no data leaks among different cross-validations
            fn_prd = pdirname + hp.weights_path + '/model_' + hp.outcome + '_' + str(idx) + '.hdf5'

            callbacks = [
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                ModelCheckpoint(fn_prd, verbose=1, save_best_only=True,save_weights_only=True) ]

            model_prd = build_model_predict(clin_params.shape[1], hp.model_layers_ae, hp.model_layers_prd)  # Build the prediction model

            # load the already-trained Encoder module
            fn_ae = pdirname + hp.weights_path + '/pretrained_model_' + hp.outcome + '_' + str(idx) + '.hdf5'
            model_prd.load_weights(fn_ae, by_name=True)

            weights = class_weight.compute_class_weight('balanced',np.unique(trv_outcomes),trv_outcomes)
            history = model_prd.fit(trv_params[train_index], trv_outcomes[train_index], batch_size=hp.batch_size,
                        epochs=hp.nr_epochs_prd, verbose=1, class_weight=weights,
                        validation_data = (trv_params[valid_index],trv_outcomes[valid_index]),
                        shuffle=True, callbacks=callbacks)

            # summarize history for loss
            plt.figure(1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train', 'valid'], loc='upper left')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')

            # load best model (before this point, the weights are those of last iteration)
            model_prd.load_weights(fn_prd) # override current weights (at last iteration) by the best performing (on validation) during training
            gt_outcomes = trv_outcomes[valid_index]
            prd_outcomes = model_prd.predict(trv_params[valid_index])

            fp, tp, thresholds = roc_curve(gt_outcomes, prd_outcomes)
            auc_ =auc(fp, tp)
            aucs.append(auc_)
            tps.append(interp(mean_fp, fp, tp))
            tps[-1][0] = 0.0

            ii = next(x for x, val in enumerate(tp) if val > 0.90)
            print('Split{:d}'.format(idx))
            print('At 0.90 Sensitivity, Specificity= {:.3f}'.format(1-fp[ii]))
            ii = next(x for x, val in enumerate(fp) if val < 0.10)
            print('Split{:d}'.format(idx))
            print('At 0.90 Specificity, Sensitivity = {:.3f}'.format(tp[ii]) )
            print('AUC = {:.3f}'.format(auc_))

            print('AUC = {:.3f}'.format(auc_))
            plt.figure(2)
            plt.plot(fp, tp, lw=1, alpha=0.3, label='AUC = {:.3f}'.format(auc_))
            plt.show()

    mean_tp = np.mean(tps, axis=0)
    mean_tp[-1] = 1.0
    mean_auc = auc(mean_fp, mean_tp)
    std_auc  = np.std(aucs)
    plt.figure(2)
    plt.plot(mean_fp, mean_tp, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tp = np.std(tps, axis=0)
    tps_upper = np.minimum(mean_tp + std_tp, 1)
    tps_lower = np.maximum(mean_tp - std_tp, 0)
    plt.fill_between(mean_fp, tps_lower, tps_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    a = hp.dataset_path
    k=[pos for pos, char in enumerate(a) if char == "/"]
    plt.title(a[k[-1]+12:-4]+'Param={:.1f} '.format(clin_params.shape[1])+' Pos={:.1f}'.format(np.sum(outcomes))+' Neg={:.1f}'.format(np.sum(1-outcomes)), fontsize=8)
    plt.legend(loc="lower right",prop={'size': 6} )
    plt.show()
