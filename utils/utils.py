import gc
import os
import platform
import psutil
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from lightgbm import LGBMClassifier as LGBMC
from sklearn.model_selection import StratifiedKFold as SKF

import torch

class Utils:
    """
    This class creates and uses several utility methods to be used across the code
    """

    def __init__(self):
        pass

    def ScoreMetric(self, ytrue, ypred)-> float:
        """
        This method calculates the metric for the competition
        Inputs- ytrue, ypred:- input truth and predictions
        Output- float:- competition metric
        """;

        score = root_mean_squared_error(ytrue, ypred)
        return score

    def clean_memory(self,):
        "Cleans unused objects and reports RAM usage"
        gc.collect()

        # Only call malloc_trim on Linux
        if platform.system() == "Linux":
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)

        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / 2.**30  # in GB
        return f"\nRAM usage = {memory_use:.4f} GB"


utils = Utils()
gc.collect()
print()

class AdversarialCVMaker:
    """
    This class assists in adversarial CV between the train and test data with the below steps-

    1. Consider any classifier as a base model, I prefer any boosted tree model as I don't have to focus too much on preprocessing
    2. Load the train and test set features
    3. Make a new target column with 1 for test set occurrances and 0 for train-set
    4. Classify to predict the test set instances with the features and new target from the step above
    
    If the AUC score hovers around 50% (random model), then we can be sure that the train and test set have similar distributions 
    Else, if our model is able to differentiate between the train and test data, then our model is unlikely to generalize as-is.
    In this case, further adjustments may be necesary 
    """

    def __init__(self, n_splits: int = 5) :
        self.model = \
        LGBMC(
            n_estimators     = 200,
            learning_rate    = 0.02,
            max_depth        = 3, 
            colsample_bytree = 0.50,
            objective        = "binary",
            metric           = "auc",
            random_state     = 42,
            device           = "gpu" if torch.cuda.is_available() else "cpu",
        )

        self.n_splits = n_splits

    @staticmethod
    def scorer(ytrue, ypreds):
        return roc_auc_score( ytrue, ypreds )

    def make_cv(
        self, Xtrain, Xtest, **fit_params,
        ):
        "Fits the model with the auxilary target and calculates the AUC score for the CV"

        df = \
        pd.concat(
            [Xtrain.assign(**{"target" : 0}), 
             Xtest.assign(**{"target" : 1}),
            ], 
            axis=0, ignore_index = True
        )

        cv     = SKF(n_splits = self.n_splits, random_state = 42, shuffle = True)
        scores = 0
        
        for train_idx, dev_idx in cv.split(df, df["target"]) :
            Xtr  = df.loc[train_idx].drop("target", axis=1)
            Xdev = df.loc[dev_idx].drop("target", axis=1)
            ytr  = df.loc[train_idx, "target"]
            ydev = df.loc[dev_idx, "target"]

            cat_cols = list( Xdev.select_dtypes(include = ["string", "category", "object"]).columns )

            if len(cat_cols) > 0 :
                Xtr[cat_cols]  = Xtr[cat_cols].astype("category")
                Xdev[cat_cols] = Xdev[cat_cols].astype("category")
            else:
                pass
                
            model = clone(self.model)
            model.fit(Xtr, ytr)
            dev_preds = model.predict_proba(Xdev)[:,1]
            score = self.scorer(ydev, dev_preds)
            scores += score

        score = scores / self.n_splits

        print(
            f"\n---> Overall adversarial CV score = {score :,.4f}"
        )

        if score > 0.60 :
            print(
                f"---> Check for test-train distribution shift\n"
            )
        else:
            print(
                f"---> Train-test distributions are similar\n"
            )