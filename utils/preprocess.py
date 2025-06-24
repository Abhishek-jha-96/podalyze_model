from gc import collect
import pandas as pd
import os

from PD_CV_LB_V1.config.config import CFG, cv_selector

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. In this case, we need to process the original data target column to be compatible with the competition dataset
    3. Check information and description
    4. Check unique values and nulls
    5. Collate starting features 
    """
    
    def __init__(self):

        self.train             = pd.read_csv(os.path.join(CFG.ip_path,"train.csv"), index_col = 'id') 
        self.test              = pd.read_csv(os.path.join(CFG.ip_path ,"test.csv"), index_col = 'id')
        self.target            = CFG.target 
        
        self.conjoin_orig_data = True if CFG.nb_orig > 0 else False
        self.dtl_preproc_req   = CFG.dtl_preproc_req
        self.test_req          = CFG.test_req
        self.cv                = cv_selector[CFG.mdlcv_mthd]
         
        self.original            = pd.read_csv(CFG.orig_path).drop_duplicates()
        self.original.index      = range(len(self.original))
        self.original.index.name = "id"    
        self.original            = self.original[self.train.columns]

        self.sub_fl = pd.read_csv(os.path.join(CFG.ip_path, "sample_submission.csv"))
        print(f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}")
        
        for tbl in [self.train, self.original, self.test]:
            obj_cols      = tbl.select_dtypes(include = ["object", "category"]).columns
            tbl.columns   = tbl.columns.str.replace(r"\(|\)|\.|\?|/|\s+","", regex = True)
            
    def _VisualizeDF(self):
        "This method visualizes the heads for the train, test and original data"
        
        print(f"\nTrain set head")
        
        print(f"\nTest set head")
        
        print(f"\nOriginal set head")
              
    def _AddSourceCol(self):
        self.train['Source']    = "Competition"
        self.test['Source']     = "Competition"
        self.original['Source'] = 'Original'
        
        self.strt_ftre = self.test.columns
        return self
        
    
    def _CollateUnqNull(self):
        
        if self.dtl_preproc_req :
            print(f"\nUnique and null values\n")
            _ = pd.concat([self.train[self.strt_ftre].nunique(), 
                           self.test[self.strt_ftre].nunique(), 
                           self.original[self.strt_ftre].nunique(),
                           self.train[self.strt_ftre].isna().sum(axis=0),
                           self.test[self.strt_ftre].isna().sum(axis=0),
                           self.original[self.strt_ftre].isna().sum(axis=0)
                          ], 
                          axis=1)
            _.columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq', 
                         'Train_Nulls', 'Test_Nulls', 'Original_Nulls'
                        ]
            
        return self
       
    def _ConjoinTrainOrig(self):
        if self.conjoin_orig_data :
            print(f"\n\nTrain shape before conjoining with original = {self.train.shape}")
            train = pd.concat([self.train] + [self.original] * CFG.nb_orig, 
                              axis=0, 
                              ignore_index = True
                             )
            print(f"Train shape after conjoining with original= {train.shape}")

            train.index = range(len(train))
            train.index.name = 'id'

        else:
            print(f"\nWe are using the competition training data only")
            train = self.train
        return train
       
    def DoPreprocessing(self):
        self._VisualizeDF()
        self._AddSourceCol()
        self._CollateInfoDesc()
        self._CollateUnqNull()
        self.train = self._ConjoinTrainOrig()

        self.train = self.train.dropna(subset = [self.target])
        self.train.index = range(len(self.train))
        
        self.cat_cols  = \
        list(
            self.test.drop("Source", axis=1).select_dtypes(["object", "string", "category"]).columns
        )
        self.cont_cols = \
        [c for c in self.strt_ftre if c not in self.cat_cols + ['Source']]
        
        return self 
            
collect()
print()