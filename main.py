from gc import collect
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from tqdm import tqdm


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder

from config.config import CFG, Mdl_Master, cv_selector
from helpers.helpers import make_ftre
from train import ModelTrainer
from utils.preprocess import Preprocessor
from utils.utils import AdversarialCVMaker, utils

def preprocess_data():
    """Preprocess the data."""
    pp = Preprocessor() 
    pp.DoPreprocessing()
    return pp


def handle_categorical_columns(Xtrain, Xtest):
    """Handle categorical columns."""
    cat_cols = Xtrain.drop("Source", axis=1).nunique()
    cat_cols = list(
        set(
            Xtrain.drop("Source", axis=1)
            .select_dtypes(["string", "object", "category"])
            .columns
        ).union(set(cat_cols.loc[cat_cols <= 500].index))
    )

    Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing")
    Xtest[cat_cols] = Xtest[cat_cols].astype("string").fillna("missing")

    return Xtrain, Xtest, cat_cols

def initialize_cv(Xtrain, Xtest, ytrain, cv):
    """Initialize the cross-validation scheme."""
    if CFG.nb_orig > 0:
        all_df = []
        for mysource in ["Competition", "Original"]:
            df = pd.concat([Xtrain.loc[Xtrain.Source == mysource], ytrain], axis=1, join="inner")
            df.index = range(len(df))
            for fold_nb, (_, dev_idx) in enumerate(cv.split(df, df[CFG.target])):
                df.loc[dev_idx, "fold_nb"] = fold_nb
            all_df.append(df)
        ygrp = pd.concat(all_df, axis=0, ignore_index=True)["fold_nb"].astype(np.uint8)
    else:
        df = Xtrain.loc[Xtrain.Source == "Competition"]
        df.index = range(len(df))
        for fold_nb, (_, dev_idx) in enumerate(cv.split(df, ytrain.iloc[df.index])):
            df.loc[dev_idx, "fold_nb"] = fold_nb
        ygrp = df["fold_nb"].astype(np.uint8)

    print(f"\n---> Shapes = {Xtrain.shape} {Xtest.shape} {ytrain.shape} {ygrp.shape}")

    return ygrp

def train_models(Xtrain, ytrain, ygrp, Xtest, cat_cols):
    """Train the models."""
    OOF_Preds = {}
    Mdl_Preds = {}
    FtreImp = {}
    drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

    for method, mymodel in tqdm(Mdl_Master.items()):
        print(f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n")

        md = ModelTrainer(
            problem_type="regression",
            es=CFG.nbrnd_erly_stp,
            target=CFG.target,
            orig_req=True if CFG.nb_orig > 0 else False,
            orig_all_folds=CFG.orig_all_folds,
            metric_lbl="rmse",
            drop_cols=drop_cols,
            pp_preds=CFG.pstprcs_oof,
        )

        sel_mdl_cols = list(Xtest.columns)
        print(f"Selected columns = {len(sel_mdl_cols):,.0f}")

        ct = ColumnTransformer(
            [("TE", TargetEncoder(random_state=CFG.state), cat_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        mypipe = Pipeline([("PP", ct), ("M", mymodel)])

        fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter = md.MakeOfflineModel(
            Xtrain,
            ytrain,
            ygrp,
            Xtest,
            mypipe,
            method,
            test_preds_req=True,
            ftreimp_plot_req=CFG.ftre_plots_req,
            ntop=50,
        )

        OOF_Preds[method] = oof_preds
        Mdl_Preds[method] = test_preds
        FtreImp[method] = ftreimp

        del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols
        print()
        collect()

    _ = utils.clean_memory()

    return OOF_Preds, Mdl_Preds

def ensemble_predictions(OOF_Preds, Mdl_Preds, Xtrain, ytrain, ygrp):
    """Ensemble the OOF predictions and test predictions."""
    len_train = Xtrain.loc[Xtrain.Source == "Competition"].shape[0]
    method = "L21R"
    model = Ridge(max_iter=10000, random_state=CFG.state)

    md = ModelTrainer(
        problem_type="regression",
        es=CFG.nbrnd_erly_stp,
        target=CFG.target,
        orig_req=False,
        orig_all_folds=CFG.orig_all_folds,
        metric_lbl="rmse",
        drop_cols=["Source", "id", "Id", "Label", CFG.target, "fold_nb"],
        pp_preds=CFG.pstprcs_oof,
    )

    _, oof_ens_preds, test_preds, _, _ = md.MakeOfflineModel(
        pd.DataFrame(OOF_Preds).iloc[0:len_train].assign(Source="Competition"),
        ytrain.iloc[0:len_train],
        ygrp.iloc[0:len_train],
        pd.DataFrame(Mdl_Preds).assign(Source="Competition"),
        model,
        method,
        test_preds_req=True,
        ftreimp_plot_req=False,
        ntop=50,
    )

    score = utils.ScoreMetric(ytrain.iloc[0:len_train], oof_ens_preds)
    print(f"\n\n---> Overall score = {score:,.8f}")


def main():
    """This is the main function that will be executed when the script is run."""

    pp = preprocess_data()

    # If adversarial CV is required, then make the CV
    if CFG.dtl_preproc_req :
        advcv = AdversarialCVMaker()
        advcv.make_cv(pp.train[pp.test.columns], pp.test)
    
    # feature transformation and CV
    Xtrain = make_ftre( pp.train.drop(CFG.target, axis=1) )
    Xtest  = make_ftre( pp.test )
    ytrain = pp.train[CFG.target]

    Xtrain, Xtest, cat_cols = handle_categorical_columns(Xtrain, Xtest)

    # Initializing the cv scheme:-
    cv = cv_selector[CFG.mdlcv_mthd]
    ygrp = initialize_cv(Xtrain, Xtest, ytrain, cv)

    OOF_Preds, Mdl_Preds = train_models(Xtrain, ytrain, ygrp, Xtest, cat_cols)

    ensemble_predictions(OOF_Preds, Mdl_Preds, Xtrain, ytrain, ygrp)

if __name__ == "__main__":
    main()