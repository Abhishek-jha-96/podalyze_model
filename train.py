import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import PredefinedSplit as PDS
from sklearn.inspection import permutation_importance
from sklearn.base import accuracy_score, clone
from sklearn.metrics import (
    cohen_kappa_score, 
    mean_absolute_error, 
    roc_auc_score, 
    root_mean_squared_error, 
    root_mean_squared_log_error
)

# TabNet models
from pytorch_tabnet.tab_model import (TabNetRegressor as TNR, TabNetClassifier as TNC)
from lightgbm import log_evaluation, early_stopping

from PD_CV_LB_V1.utils import utils


def MakePermImp(
        method, mdl, X, y, ygrp,
        myscorer, 
        n_repeats = 2,
        state = 42,
        ntop: int = 15,
        **params,
):
    """
    This function makes the permutation importance for the provided model and returns the importance scores for all features
    
    Note-
    myscorer - scikit-learn -> metrics -> make_scorer object with the corresponding eval metric and relevant details
    """

    cv        = PDS(ygrp)
    n_splits  = ygrp.nunique()
    drop_cols = ["Source", "id", "Id", "Label", "fold_nb"]

    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
        Xtr  = X.iloc[train_idx].drop(drop_cols, axis=1, errors = "ignore")
        Xdev = X.iloc[dev_idx].drop(drop_cols, axis=1, errors = "ignore")
        ytr  = y.loc[Xtr.index]
        ydev = y.loc[Xdev.index]

        model = clone(mdl)
        sel_cols = list(Xdev.columns)
        model.fit(Xtr, ytr)

        imp_ = permutation_importance(model,
                                      Xdev, ydev,
                                      scoring = myscorer,
                                      n_repeats = n_repeats,
                                      random_state = state,
                                      )["importances_mean"]
        imp_ = pd.Series(index = sel_cols, data = imp_)

        return imp_

class ModelTrainer:
    "This class trains the provided model on the train-test data and returns the predictions and fitted models"

    def __init__(
        self,
        problem_type   : str   = "binary", 
        es             : int   = 100,
        target         : str   = "",
        metric_lbl     : str   = "auc",
        orig_req       : bool  = False,
        orig_all_folds : bool  = False,
        drop_cols      : list  = ["Source", "id", "Id", "Label", "fold_nb"],
        pp_preds       : bool  = False,
    ):
        """
        Key parameters-
        es_iter  - early stopping rounds for boosted trees
        pp_preds - do you want to post-process predictions (true/ false boolean)
        """

        self.problem_type   = problem_type
        self.es_iter        = es
        self.target         = target
        self.drop_cols      = drop_cols + [self.target]
        self.metric_lbl     = metric_lbl
        self.orig_req       = orig_req
        self.orig_all_folds = orig_all_folds
        self.pp_preds       = pp_preds

        if self.metric_lbl == "rmse" :
            self.ScoreMetric = lambda x,y : root_mean_squared_error( x, self.PostProcessPreds( y))
        elif self.metric_lbl == "mae" :
            self.ScoreMetric = lambda x,y : mean_absolute_error( x, self.PostProcessPreds( y))
        elif self.metric_lbl == "accuracy" :
            self.ScoreMetric = lambda x,y : accuracy_score( np.uint8( x ), np.uint8(self.PostProcessPreds( y ) ) )
        elif self.metric_lbl == "auc" :
            self.ScoreMetric = lambda x,y : roc_auc_score(  x , self.PostProcessPreds( y )  )
        elif self.metric_lbl == "kappa" :
            self.ScoreMetric = lambda x,y : cohen_kappa_score( np.uint8( x ), np.uint8(self.PostProcessPreds( y ) ), 
                                                              weights = "quadratic" 
                                                             )
        elif self.metric_lbl == "rmsle" :
            self.ScoreMetric = lambda x,y : root_mean_squared_log_error( x, self.PostProcessPreds( y))
        else:
            self.ScoreMetric = utils.ScoreMetric


    def PostProcessPreds(self, ypred):
        "This method post-processes predictions optionally"
        if self.pp_preds :
            return np.clip(ypred, a_min = 0, a_max = 1)
        else:
            return ypred
            
    def LoadData(
            self, X, y, Xtest,
            train_idx : list = [],
            dev_idx   : list = [],
            ):
        "This method loads the train and test data for the model fold using/ not using the original data"

        try:
            mysrc = X["Source"]
        except:
            X["Source"] , Xtest["Source"] = ("Competition", "Competition")

        if self.orig_req == False:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        elif self.orig_req == True and self.orig_all_folds == True:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

            orig_x = X.query("Source == 'Original'")[Xtr.columns]
            orig_y = y.iloc[orig_x.index]

            Xtr = pd.concat([Xtr, orig_x], axis = 0, ignore_index = True)
            ytr = pd.concat([ytr, orig_y], axis = 0, ignore_index = True)

        elif self.orig_req == True and self.orig_all_folds == False:
            Xtr  = X.iloc[train_idx].drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        Xt = Xtest[Xdev.columns]

        print(f"\n---> Shapes = {Xtr.shape} {ytr.shape} -- {Xdev.shape} {ydev.shape} -- {Xt.shape}")
        return (Xtr, ytr, Xdev, ydev, Xt)
    
    def MakePreds(self, X, fitted_model):
        "This method creates the model predictions based on the model provided, with optional post-processing"

        if self.problem_type == "regression":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict(X.to_numpy()).flatten())
            else:
                return self.PostProcessPreds(fitted_model.predict(X))
                
        elif self.problem_type == "binary":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict_proba(X.to_numpy()[:,1]).flatten())
            else:
                return self.PostProcessPreds(fitted_model.predict_proba(X)[:, 1])
                
        elif self.problem_type == "multiclass":
            if isinstance(fitted_model, (TNC, TNR)) == True:
                return self.PostProcessPreds(fitted_model.predict_proba(X.to_numpy()))
            else:
                return self.PostProcessPreds(fitted_model.predict_proba(X))

    def MakeOrigPreds(
            self, orig: pd.DataFrame, fitted_models: list, n_splits : int, ygrp: pd.Series,
            ):
        "This method creates the original data predictions separately only if required"

        if self.orig_req == False:
            orig_preds = 0

        elif self.orig_req == True and self.orig_all_folds == True:
            orig_preds = 0
            df = orig.drop(self.drop_cols, axis = 1, errors = "ignore")

            for fitted_model in fitted_models:
                orig_preds = orig_preds + (self.MakePreds(df, fitted_model) / n_splits)

        elif self.orig_req == True and self.orig_all_folds == False:
            len_orig   = orig.shape[0]
            orig.index = range(len_orig)
            orig_ygrp  = ygrp[-1 * len_orig:]
            orig_ygrp.index = range(len_orig)
            
            orig_preds = np.zeros(len_orig)
            for fold_nb, fitted_model in enumerate(fitted_models):
                df = \
                orig.iloc[orig_ygrp.loc[orig_ygrp == fold_nb].index].\
                drop(self.drop_cols, axis=1, errors = "ignore")
                
                orig_preds[df.index] = self.MakePreds(df, fitted_model)
                del df
        return orig_preds

    def MakeOfflineModel(
        self, X, y, ygrp, Xtest, mdl, method,
        test_preds_req   : bool = True,
        ftreimp_plot_req : bool = True,
        ntop             : int  = 50,
        **params,
    ):
        """
        This method trains the provided model on the dataset and cross-validates appropriately

        Inputs-
        X, y, ygrp       - training data components (Xtrain, ytrain, fold_nb)
        Xtest            - test data (optional)
        model            - model object for training
        method           - model method label
        test_preds_req   - boolean flag to extract test set predictions
        ftreimp_plot_req - boolean flag to plot tree feature importances
        ntop             - top n features for feature importances plot

        Returns-
        oof_preds, test_preds - prediction arrays
        fitted_models         - fitted model list for test set
        ftreimp               - feature importances across selected features
        mdl_best_iter         - model average best iteration across folds
        """

        oof_preds     = np.zeros(len(X.loc[X.Source == "Competition"]))
        orig_preds    = np.zeros(len(X.loc[X.Source == "Original"]))
        test_preds    = []
        mdl_best_iter = []
        ftreimp       = 0

        scores, tr_scores, fitted_models = [], [], []

        if self.orig_req == True:
            cv = PDS(ygrp)
        elif self.orig_req == False:
            X  = X.loc[X.Source == "Competition"]
            y  = y.iloc[X.index]
            cv = PDS(ygrp.iloc[0 : len(X)])

        n_splits = ygrp.nunique()

        for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
            Xtr, ytr, Xdev, ydev, Xt = \
            self.LoadData(X, y, Xtest, train_idx, dev_idx)

            model = clone(mdl)

            if "CB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose = 0,
                          early_stopping_rounds = self.es_iter,
                          )
                best_iter = model.get_best_iteration()

            elif "LGB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          callbacks = [log_evaluation(0),
                                       early_stopping(stopping_rounds = self.es_iter, verbose = False,),
                                       ],
                          eval_metric = self.metric_lbl,
                          )
                best_iter = model.best_iteration_

            elif "XGB" in method and self.es_iter > 0:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose  = 0,
                          )
                best_iter = model.best_iteration

            else:
                model.fit(Xtr, ytr)
                best_iter = -1

            fitted_models.append(model)

            try:
                ftreimp += model.feature_importances_
            except:
                try:
                    ftreimp += model.coef_.flatten()
                except:
                    pass

            try:
                ftreimp += model["M"].feature_importances_
            except:
                try:
                    ftreimp += model["M"].coef_.flatten()
                except:
                    pass               
            
            dev_preds = self.MakePreds(Xdev, model)
            oof_preds[Xdev.index] = dev_preds

            train_preds  = self.MakePreds(Xtr, model)
            tr_score     = self.ScoreMetric(ytr.values.flatten(), train_preds)
            score        = self.ScoreMetric(ydev.values.flatten(), dev_preds)

            scores.append(score)
            tr_scores.append(tr_score)

            nspace = 15 - len(method) - 2 if fold_nb <= 9 else 15 - len(method) - 1

            if self.es_iter > 0 :
                print(f"{method} Fold{fold_nb} {' ' * nspace} OOF = {score:.6f} | Train = {tr_score:.6f} | Iter = {best_iter:,.0f} ")
            else:
                print(f"{method} Fold{fold_nb} {' ' * nspace} OOF = {score:.6f} | Train = {tr_score:.6f} ")
                
            mdl_best_iter.append(best_iter)

            if test_preds_req:
                test_preds.append(self.MakePreds(Xt, model))
            else:
                pass

        test_preds    = np.mean(np.stack(test_preds, axis = 1), axis=1)
        ftreimp       = pd.Series(ftreimp, index = Xdev.columns)
        mdl_best_iter = np.uint16(np.amax(mdl_best_iter))

        print(f"\n---> {np.mean(scores):.6f} +- {np.std(scores):.6f} | OOF")
        print(f"---> {np.mean(tr_scores):.6f} +- {np.std(tr_scores):.6f} | Train")

        if self.es_iter <= 0 :
            pass
        else:
            print(
                f"---> Max best iteration = {mdl_best_iter :,.0f}"
            )

        if self.orig_req:
            print(f"---> Collecting original predictions")
            orig_preds = self.MakeOrigPreds(X.loc[X.Source == "Original"],
                                            fitted_models,
                                            n_splits,
                                            ygrp,
                                            )
            oof_preds = np.concatenate([oof_preds, orig_preds], axis= 0)
        else:
            pass
        return (fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter)

    def MakeOnlineModel(
        self, X, y, Xtest, model, method,
        test_preds_req : bool = False,
    ):
        "This method refits the model on the complete train data and returns the model fitted object and predictions"

        try:
            model.early_stopping_rounds = None
        except:
            pass

        if "TN" in method:
            model.fit(
                X.to_numpy(), y.to_numpy().reshape(-1,1),
                max_epochs  = 100,
                batch_size  = 128,
                virtual_batch_size = 64,
                )
        else:
            try:
                model.fit(X, y, verbose = 0)
            except:
                model.fit(X, y,)

        oof_preds  = self.MakePreds(X, model)
        if test_preds_req:
            test_preds = self.MakePreds(Xtest[X.columns], model)
        else:
            test_preds = 0
            
        return (model, oof_preds, test_preds)

class HillClimber:
    "This class develops the Hill Climber algorithm for the provided datasets"

    def __init__(self):
        self.ScoreMetric = utils.ScoreMetric

    def DoHillClimb(
        self, 
        target:str,
        direction:str,
        cutoff:float,
        neg_wgt:str,
        OOF_Preds: pd.DataFrame,
        Mdl_Preds: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ):
        """
        This method performs hill-climbing on the OOF and Test predictions dataset and returns the below-
        1. OOF ensemble predictions
        2. Test set predictions
        3. Score dataframe (with scores in sort-order)
        """

        oof_df     = OOF_Preds
        test_preds = Mdl_Preds
    
        # Scoring the individual models:-
        Scores = pd.DataFrame(index = oof_df.columns, columns = ['Score'])
    
        for col in oof_df.columns:
            Scores.at[col, 'Score'] = self.ScoreMetric(y, oof_df[col].values.flatten())
    
        # Sorting scores
        Scores.sort_values(
            by= 'Score',
            ascending = [True if direction == 'minimize' else False],
            inplace = True,
        )
    
        print(f"\n ----- Initiating hill-climb ----- \n");
        STOP = False
        current_best_ensemble   = oof_df.iloc[:,0]
        current_best_test_preds = test_preds.iloc[:,0]
        MODELS                  = oof_df.iloc[:,1:]
    
        if neg_wgt == "Y":
            weight_range = np.arange(-0.5,0.51,0.01);
        else:
            weight_range = np.arange(0.01,0.51,0.01);
    
        history = [self.ScoreMetric(y, current_best_ensemble)]
    
        i=0
    
        # Hill climbing algorithm:-
        while not STOP:
            i+=1
    
            potential_new_best_cv_score = self.ScoreMetric(y, current_best_ensemble)
            k_best, wgt_best = None, None
    
            for k in MODELS:
                for wgt in weight_range:
                    potential_ensemble = (1- wgt) * current_best_ensemble + wgt * MODELS[k]
                    cv_score = self.ScoreMetric(y, potential_ensemble)
    
                    if direction == 'minimize':
                        if cv_score < potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
                    if direction == 'maximize':
                        if cv_score > potential_new_best_cv_score:
                            potential_new_best_cv_score, k_best, wgt_best = cv_score, k, wgt
    
            if k_best is not None:
                current_best_ensemble   = (1- wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
                current_best_test_preds = (1- wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
                MODELS.drop(k_best, axis=1, inplace=True)
    
                if MODELS.shape[1]==0:  STOP = True
    
                num_space = 50 - len(k_best) if i <= 9 else 49 - len(k_best)
                print(f" {i}.{k_best} {' ' * num_space} Weight = {wgt_best: .4f} {' ' * 5} Score = {potential_new_best_cv_score:.6f}")
                del num_space
    
                history.append(potential_new_best_cv_score)
    
            else:
                STOP = True
    
        return (current_best_ensemble, current_best_test_preds, Scores)