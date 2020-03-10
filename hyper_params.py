import numpy as np
from collections import namedtuple

HParams = namedtuple(
    "HParams",
    [
        # Architecture: number of nodes
	    "model_layers_ae",
        "model_layers_prd",
        # Application
        "outcome",
        #training
    	"batch_size",
        "nr_epochs_ae",
        "nr_epochs_prd",
        "op_mode",
        "xrsval_model_to_use",

        #Directories/Storage
        "weights_path",
        "dataset_path",
        "results_path"
    ])

def create_hparams():
  cohort = 'LVOTO' # LVOTO or  nonLVOTO
  return HParams(
      op_mode ='train', # pretrain, or train (already include test)!
      xrsval_model_to_use = 7, # corresponding to best 'validation model': LVOTO=7  nonLVOTO=4

      model_layers_ae = np.asarray( [8, 4, 2] ),
      model_layers_prd= np.asarray( [8, 1] ),
      # training
      batch_size    = 25,
      nr_epochs_ae  = 80,
      nr_epochs_prd = 100,

      # Outcome used for classification (NN output)
      outcome='HF',
      # Directories/Storage
      weights_path = '/models/'+cohort,
      dataset_path='/datasets/refined_cohort_' + cohort + '_HF_6month.mat',
      results_path = '/results/'+cohort
  )