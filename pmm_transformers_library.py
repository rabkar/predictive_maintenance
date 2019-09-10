import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pmm_tools_function import mark_timestamp

pd.options.mode.chained_assignment = None  # default='warn'

class EnsureDataTypes():

    def __init__(self, features, dtypes=None):
        self.features = features
        self.dtypes = dtypes
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.dtypes is None:
            X.loc[:, self.features] = X.loc[:, self.features].astype(np.double)
        return X

class HealthScoreModelRouter():
    
    def __init__(self, flag_columns, routes):
        self.flag_columns = flag_columns
        self.routes = routes
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Xcopy = X.copy()
        for flag, model, features in self.routes:
            idx = X[X[self.flag_columns]==flag].index
            if len(idx)>0:
                Xcopy.loc[idx, 'health_score'] = model.predict_proba(X.loc[idx, features])[:,0]
        result = Xcopy[['health_score', self.flag_columns]].values
        
        return result            
        

class PapOilImputation():
            
    def map_unit_hours(self, x):
        x = int(x)
        if x > self.standard_lifetime:
            return int(x - int(x/self.standard_lifetime)*self.standard_lifetime)
        else:
            return int(x)

    def __init__(self, unit_model, component, features, 
                 unit_hours_bin=500, standard_lifetime=20000):
        self.unit_model=unit_model
        self.features=features
        self.component=component
        self.unit_hours_bin=unit_hours_bin
        self.standard_lifetime=standard_lifetime
    
    def fit(self, X, y=None):
        print("{}: Fitting Oil Analyisis data for references".format(mark_timestamp()))
        X = X[(X['MODL_NUM']==self.unit_model) & 
              (X['COMPONENT']==self.component) &
              (X['HRS_KM_TOT']<self.standard_lifetime) &
              (X['HRS_KM_TOT']>=self.unit_hours_bin)
             ][['HRS_KM_TOT']+self.features]
        X['UNIT_HOURS_GROUP'] = X['HRS_KM_TOT'].map(
            lambda x: int(self.map_unit_hours(x)/self.unit_hours_bin)*self.unit_hours_bin)
        self.reference = X.groupby(['UNIT_HOURS_GROUP'])[self.features].mean()
        return self
    
    def transform(self, X, y=None):
        for c in self.features:
            if c in X.columns:
                abnormal_idx = X[X[c].isnull()].index
                ref_idx = np.array(
                    X.loc[abnormal_idx]['HRS_KM_TOT'].map(
                        lambda x: int(self.map_unit_hours((x))/self.unit_hours_bin)*self.unit_hours_bin
                                if self.map_unit_hours(x)>self.unit_hours_bin else self.unit_hours_bin))
                X.loc[abnormal_idx, c] = [self.reference.loc[i, c] for i in ref_idx]
            else:
                X[c] = np.nan
        return X


class PapOilDataCleanser():
    
    def __init__(self, unit_model, component, features):
        self.component = component
        self.unit_model = unit_model
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X[(X['MODL_NUM']==self.unit_model) & (X['COMPONENT']==self.component) & (X['HRS_KM_OC']>0)]
        X['SRL_NUM'] = X['SRL_NUM'].astype(str)
        X['SAMPL_DT'] = X['SAMPL_DT'].astype(str)
        return X[['LAB_NUM', 'SRL_NUM', 'MODL_NUM', 'COMPONENT', 'HRS_KM_OC', 'HRS_KM_TOT', 'SAMPL_DT']+self.features] 

class RemoveByThreshold():

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

class InverseTransfromStandardScaller():
    
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
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X.loc[:,self.features] = self.dfStandardScaler.standardscaler\
                .inverse_transform(X[self.features])
        return X

class FeaturesSelector():

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.features]

class DfStandardScaler():

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
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X.loc[:,self.features] = self.standardscaler.transform(X[self.features])
        return X

class DfMapMinMaxScaler():

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


class DeriveFeatures():

    def __init__(self, functions):
        self.functions = functions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if len(self.functions) > 1:
            for f in self.functions:
                f(X)
        else:
            self.functions[0](X)
        return X

class EquipmentSelector():

    def __init__(self, equipment_list):
        self.equipment_list = equipment_list
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        equipment_idx = X[~X['UNIT_SRL_NUM'].isin(self.equipment_list)]
        X.drop(equipment_idx, inplace=True)
        return X
    
class VHMSReplaceSensorErrorValue():
    
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        print("{}: Fitting sensor error scaler with data".format(mark_timestamp()))
        self.equipment_catalogue = X['UNIT_SRL_NUM'].drop_duplicates().tolist()
        self.compute_mean_by_serial_number(X)
        print("\t{}: Finish fitting scaler".format(mark_timestamp()))
        return self
    
    def transform(self, X, y=None):
        print("{}: Transforming data".format(mark_timestamp()))
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
            
        print("\t{}: Replacing error value with average for each serial number".format(mark_timestamp()))
        for c in self.features:
            if c in X.columns:
                # find index which data is anomaly
                abnormal_idx = X[(X[c].isnull()) | (X[c]<0) | (X[c]>1E4)].index
                # replace abnormal datum with mean value
                X.loc[abnormal_idx, c] = X.loc[abnormal_idx, 'UNIT_SRL_NUM']\
                    .map(lambda x: self.get_equipment_average(x).loc[c] 
                        if equipment_exist(x) else self.all_equipment_average.loc[c])
            else:
                X[c] = self.all_equipment_average.loc[c]
            X[c] = X[c].astype(np.double)
        return X
    
    def replace_error_with_nan(self, X):
        print("\t{}: Separate sensor value from mean calculation".format(mark_timestamp()))
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
        print("\t{}: Computing average value of each equipment".format(mark_timestamp()))
        for srl_num in srl_num_list:
            Xsub = Xcopy[Xcopy['UNIT_SRL_NUM']==srl_num][self.features]
            self.equipment_average[srl_num] = Xsub.mean()
        print("\t{}: Computing average value of all equipment".format(mark_timestamp()))
        self.all_equipment_average = Xcopy[self.features].mean()
        del Xcopy
