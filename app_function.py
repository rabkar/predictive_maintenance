import numpy as np
import pandas as pd
import json
import joblib
from ut_pdm_tools_lib import join_vhms_with_pap, make_smooth, from_pandas_to_json
from datetime import datetime, timedelta

def add_response_identity(data):
    data["__dt"] = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    data["__ts"] = int((datetime.now() - datetime(1970,1,1)).total_seconds())
    # add other required fields if necessary
    # example data["requester"] = "CBM" or data["requester"] = "Customer Portal"
    return data

def validate_data(data):
    # function to validate data 
    if type(data)==dict:
        return [data]
    elif type(data)==list:
        return data
    else:
        return None

def calculate_health_score(request):
    # parse request to get data
    data = request.get_json(force=True)
    if type(data)==str:
        data = json.loads(data)
    unit_model = data.get("unit_model")
    component = data.get("component")

    # validate data vhms
    vhms = data.get("vhms")
    vhms = pd.DataFrame(validate_data(vhms))
    vhms['UNIT_MODL'] = unit_model.upper()

    # validate data pap
    pap = data.get("pap")
    pap = validate_data(pap)
    if pap is not None and len(pap)>0:
        pap = pd.DataFrame(validate_data(pap))
    else:
        pap = None

    # load trained data-preparation pipeline and machine learning model
    model_id = unit_model.lower() + "_" + component.lower()
    vhms_pipe = joblib.load('model/{}_vhms_prep_pipe.pkl'.format(model_id))
    pap_pipe = joblib.load('model/{}_pap_prep_pipe.pkl'.format(model_id))
    hs_scoring_pipe = joblib.load('model/{}_health_scoring_pipe.pkl'.format(model_id))
    
    # prepare vhms and pap data before scoring
    vhms_transform = vhms_pipe.transform(vhms)
    if pap is not None:
        pap_transform = pap_pipe.transform(pap)
        scoring_dataset = join_vhms_with_pap(vhms_transform, pap_transform, time_window=30)
        scoring_dataset['with_pap'] = scoring_dataset['LAB_NUM'].map(
            lambda x: True if x is not None and x==x else False)
    else:
        scoring_dataset = vhms_transform.copy()
        scoring_dataset['with_pap'] = False
        scoring_dataset['LAB_NUM'] = None
    
    # compute health score
    hs_result = hs_scoring_pipe.transform(scoring_dataset)
    hs = hs_result[:,0].astype(np.double)
    hs = make_smooth(hs,window_size=7)
    
    # store and return the result
    result_dataset = pd.DataFrame({
        "serial_number": scoring_dataset['UNIT_SRL_NUM'],
        "smr": scoring_dataset["SMR"],
        "timestamp": scoring_dataset['TIMESTAMP'],
        "health_score": hs,
        "pap_ref_lab_num": scoring_dataset['LAB_NUM']
    })    
    health_score_result = from_pandas_to_json(result_dataset)
    response = {
        # data header. add underscore ("_") so that it appears above alphabethically
        "_unit_model": unit_model.upper(),
        "_component": component,
        # data content
        "health_score_data": health_score_result
    }
    return response
