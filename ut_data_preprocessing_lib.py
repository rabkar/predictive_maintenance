import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ut_pdm_tools_lib import CommonFunctions

pd.options.mode.chained_assignment = None  # default='warn'

class RemoveByThreshold(TransformerMixin):

    def __init__(self, features, threshold):
        self.features = features
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        drop_indexes = []
        for i, c in enumerate(self.features):
            upper_thres = self.threshold[i][1]
            lower_thres = self.threshold[i][0]
            drop_indexes += X[(X[c]>upper_thres) | (X[c]<lower_thres)].index.tolist()
        drop_indexes = list(set(drop_indexes))
        X.drop(drop_indexes, axis=0, inplace=True)
        return X

class InverseTransfromStandardScaller(TransformerMixin):
    
    def __init__(self, dfStandardScaler):
        self.dfStandardScaler = dfStandardScaler
        self.features = dfStandardScaler.features
    
    def fit(self, X, y=None):
        self.mean = self.dfStandardScaler.mean
        self.scale = self.dfStandardScaler.scale
        self.var = self.dfStandardScaler.var
        self.n_sample = self.dfStandardScaler.n_sample
        return self
    
    def transform(self, X, y=None):
        X.loc[:,self.features] = self.dfStandardScaler.standardscaler.inverse_transform(X[self.features])
        return X

class FeaturesSelector(BaseEstimator):

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.features]

class DfStandardScaler(BaseEstimator):

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.standardscaler = StandardScaler()
        self.standardscaler.fit(X[self.features])
        self.mean = self.standardscaler.mean_
        self.scale = self.standardscaler.scale_
        self.var = self.standardscaler.var_
        self.n_sample = self.standardscaler.n_samples_seen_
        return self
    
    def transform(self, X, y=None):
        X.loc[:,self.features] = self.standardscaler.transform(X[self.features])
        return X

class DfMapMinMaxScaler(MinMaxScaler):

    def __init__(self, features, feature_range=(0,1)):
        self.features = features
        self.feature_range = feature_range
    
    def fit(self, X, y=None):
        self.minmaxscaler = MinMaxScaler(feature_range=self.feature_range)
        self.minmaxscaler.fit(X[self.features])
        return self
    
    def transform(self, X, y=None):
        X.loc[:,self.features] = self.minmaxscaler.transform(X[self.features])
        return X


class DeriveInteractionFeatures(BaseEstimator):

    def __init__(self, expressions):
        self.expressions = expressions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for e in self.expressions:
            print("{}: Deriving expression ".format(CommonFunctions.mark_timestamp()))
            print("\t{}".format(e))
            exec(e)
        return X

class EquipmentSelector(BaseEstimator):

    def __init__(self, equipment_list):
        self.equipment_list = equipment_list
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        equipment_idx = X[~X['UNIT_SRL_NUM'].isin(self.equipment_list)]
        X.drop(equipment_idx, inplace=True)
        return X
    
class VHMSReplaceSensorErrorValue(BaseEstimator):
    
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        print("{}: Fitting sensor error scaler with data".format(CommonFunctions.mark_timestamp()))
        self.compute_mean_by_serial_number(X)
        self.equipment_catalogue = X['UNIT_SRL_NUM'].drop_duplicates().tolist()
        print("\t{}: Finish fitting scaler".format(CommonFunctions.mark_timestamp()))
        return self
    
    def transform(self, X, y=None):
        print("{}: Transforming data".format(CommonFunctions.mark_timestamp()))
        self.replace_error_with_average_serial_number(X)
        return X
    
    def get_equipment_average(self, srl_num):
        if srl_num in self.equipment_catalogue:
            return self.equipment_average[srl_num]
        else:
            return None
        
    def replace_error_with_average_serial_number(self, X):
        def equipment_exist(srl_num):
            if srl_num in self.equipment_catalogue:
                return True
            else:
                return False
            
        # z = X[[c for c in self.features if c in X.columns]].values
        print("\t{}: Replacing error value with average for each serial number".format(CommonFunctions.mark_timestamp()))
        for c in self.features:
            if c in X.columns:
                abnormal_idx = X[(X[c]<0) | (X[c]>1E4)].index
                X.loc[abnormal_idx, c] = X.loc[abnormal_idx, 'UNIT_SRL_NUM']\
                    .map(lambda x: self.get_equipment_average(x).loc[c] 
                        if equipment_exist(x) else self.all_equipment_average.loc[c])
            else:
                X[c] = np.nan
        return X
    
    def replace_error_with_nan(self, X):
        print("\t{}: Separate sensor value from mean calculation".format(CommonFunctions.mark_timestamp()))
        # z = X[[c for c in self.features if c in X.columns]].values
        for c in self.features:
            if c in X.columns:
                abnormal_idx = X[(X[c]<0) | (X[c]>1E4)].index
                if len(abnormal_idx) > 0:
                    X.loc[abnormal_idx, c] = np.nan
            else:
                X[c] = np.nan
        return X
    
    def compute_mean_by_serial_number(self, X):
        Xcopy = X.copy()
        self.replace_error_with_nan(Xcopy)
        self.equipment_average = {}
        srl_num_list = Xcopy['UNIT_SRL_NUM'].drop_duplicates().tolist()
        print("\t{}: Computing average value of each equipment".format(CommonFunctions.mark_timestamp()))
        for srl_num in srl_num_list:
            Xsub = Xcopy[Xcopy['UNIT_SRL_NUM']==srl_num][self.features]
            self.equipment_average[srl_num] = Xsub.mean()
        print("\t{}: Computing average value of all equipment".format(CommonFunctions.mark_timestamp()))
        self.all_equipment_average = Xcopy[self.features].mean()
        del Xcopy
