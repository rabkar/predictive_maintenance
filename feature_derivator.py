import numpy as np
from ut_pdm_tools_lib import join_vhms_with_pap, logistic_function

def engine_hd785_feature_derivator(X):
    # interaction features
    X['FUEL_PER_ENGINE_CYCLE'] = X['FUEL_RATE']*1E6/(60*X['ENG_SPEED_AV'])
    X['COOL_TEMP_GRADIENT'] = 2**(X['ENG_OIL_TMP_MX']-X['COOL_TEMP_MX'])
    X['ENG_SPEED_RANGE'] = X['ENG_SPEED_MX'] - X['ENG_SPEED_AV']
    X['RPM_TO_SPEED_RATIO'] = np.log(X['ENG_SPEED_MX'].tolist())/np.log(X['TRAVELSPEED_MX'].tolist())
    
    # single-parameter-transformation
    X['COOL_TEMP_MX_LOG'] = X['COOL_TEMP_MX'].map(lambda x: np.exp(x/15))
    X['ENG_OIL_TMP_MX_LOG'] = X['ENG_OIL_TMP_MX'].map(lambda x: np.exp(x/15))
    X['BLOWBY_PRESS_MX_LOG'] = X['BLOWBY_PRESS_MX'].map(lambda x: logistic_function(x, k=0.35, m=4.5))


def engine_pc2000_feature_derivator(X):
    # interaction features
    X['FUEL_PER_ENGINE_CYCLE'] = X['FUEL_RATE']*1E6/(60*X['ENGSPEED_AV'])
    X['COOL_TEMP_GRADIENT'] = X['ENGOIL_TMPMAX']-X['COOL_TEMPMAX']
    X['ENG_SPEED_RANGE'] = X['ENGSPEED_MX'] - X['ENGSPEED_AV']
    X['POWER_PER_CYCLE_AV'] = X['ENG_PWR_AV']/X['ENGSPEED_AV']
    X['POWER_PER_CYCLE_MX'] = X['ENG_PWR_MX']/X['ENGSPEED_MX']
    X['ENGINE_CONSTANT_MAX_'] = ((X['PUMP_1_TORQUE_MX'] + X['PUMP_2_TORQUE_MX'])*X['ENGSPEED_MX'])/(X['ENG_PWR_MX']*1E5)
    X['ENGINE_CONSTANT_AVE_'] = ((X['PUMP_1_TORQUE_AV'] + X['PUMP_2_TORQUE_AV'])*X['ENGSPEED_AV'])/(X['ENG_PWR_AV']*1E5)
    
    # single-parameter-transformation
    X['COOL_TEMP_MX_LOG'] = X['COOL_TEMPMAX'].map(lambda x: np.exp((x-100)/6.5))
    X['COOL_TEMP_GRADIENT'] = X['COOL_TEMP_GRADIENT'].map(lambda x: np.exp((x-10-2)/6.5))
    X['ENG_OIL_TMP_MX_LOG'] = X['ENGOIL_TMPMAX'].map(lambda x: np.exp((x-110)/5.5))
    X['BLOWBY_PRESS_MX_LOG'] = X['BLOWBYPRESS_MX'].map(lambda x: np.exp((x-20)/5))
    X['ENGINE_CONSTANT_MAX'] = X['ENGINE_CONSTANT_MAX_']#.map(lambda x: np.exp(-50*(x-0.18)/0.5))
    X['ENGINE_CONSTANT_AVE'] = X['ENGINE_CONSTANT_AVE_']#.map(lambda x: np.exp(-100*(x-0.09)/0.7))