from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import KFold, GroupKFold as GKF, RepeatedKFold as RKF, RepeatedStratifiedKFold as RSKF
from catboost import CatBoostRegressor as CBR
from xgboost import XGBRegressor as XGBR

import torch

class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb         = 1
    model_id           = "V1_3"
    model_label        = "ML"
    test_req           = False
    test_iter          = 50
    gpu_switch         = "ON" if torch.cuda.is_available() else "OFF"
    state              = 42
    target             = f"Listening_Time_minutes"
    grouper            = f""
    tgt_mapper         = {}
    ip_path            = f"/kaggle/input/playground-series-s5e4" # Change this to input path
    op_path            = f"/kaggle/working" # Change this to output path
    orig_path          = f"/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv" # Change this to original dataset path
    data_path          = f""
    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True
    nb_orig            = 1
    orig_all_folds     = True

    # Model Training:-
    pstprcs_oof        = False
    pstprcs_train      = False
    pstprcs_test       = False
    ML                 = True
    test_preds_req     = True
    n_splits           = 5
    n_repeats          = 1
    nbrnd_erly_stp     = 0
    mdlcv_mthd         = 'KF'
    metric_obj         = 'minimize'


cv_selector = {
 "RKF"   : RKF(n_splits   = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits  = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits   = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits   = CFG.n_splits)
}

Mdl_Master = {   
    f'CB1R'    : CBR(**{"loss_function"         : "RMSE",
                        "eval_metric"           : "RMSE",
                        'task_type'             : "GPU" if CFG.gpu_switch == "ON" else "CPU",
                        'learning_rate'         : 0.0225,
                        'iterations'            : 4_500 if CFG.test_req == False else 50,
                        'max_depth'             : 6,
                        'min_data_in_leaf'      : 39 ,
                        'colsample_bylevel'     : 0.55 if CFG.gpu_switch == "OFF" else None,
                        'l2_leaf_reg'           : 2.50,
                        'random_strength'       : 0.025,
                        'leaf_estimation_method': "Newton",
                        'od_wait'               : 25, 
                        'verbose'               : 0,
                        'random_state'          : CFG.state,
                        }
                    ),
        
    f'XGB1R'  : XGBR(**{  "objective"            : "reg:squarederror",
                        "eval_metric"          : "rmse",
                        'device'               : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                        'learning_rate'        : 0.02,
                        'n_estimators'         : 4_500 if CFG.test_req == False else 50,
                        'max_depth'            : 5,
                        'colsample_bytree'     : 0.60,
                        'colsample_bynode'     : 0.65,
                        'subsample'            : 0.65,
                        'reg_lambda'           : 0.001,
                        'reg_alpha'            : 0.001,
                        'verbosity'            : 0,
                        'random_state'         : CFG.state,
                        'early_stopping_rounds': None if CFG.nbrnd_erly_stp == 0 else CFG.nbrnd_erly_stp,
                        } 
                    ),
    }