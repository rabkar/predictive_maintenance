from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from pandasql import sqldf
import numpy as np

def estimate_rul(smr, hs, midlife_smr=6000, hs_limit=0.05, window_size=5000):
    # function to estimate rul and its supporting insight
    # rul: remaining useful lifetime
    # eol: end of lifetime (rul - current_smr)

    def rul_correction(eol, smr):
        # sub-function to correct rul and eol if eol is before smr, then return NaN instead
        if eol < smr:
            eol = np.nan
            rul = np.nan
        else:
            rul = eol - smr
            if rul > 30000:
                eol = np.nan
                rul = np.nan
        return (eol, rul)
        
    max_smr = smr.max()
    min_smr = smr.min()
    grad_ex, itcp_ex = linear_regression(smr, hs)
    eol_1 = (hs_limit - itcp_ex)/grad_ex
    (eol_1, rul_1) = rul_correction(eol_1, max_smr)
    
    maxima, minima = locate_extreme(y, window_size=25)
    if len(minima) > 2:
        smr_minima = x[minima]
        hsc_minima = y[minima]
        grad_mn, itcp_mn = linear_regression(smr_minima, hsc_minima)
        eol_2 = (hs_limit - itcp_mn)/grad_mn
        (eol_2, rul_2) = rul_correction(eol_2, max_smr)
    else:
        (eol_2, rul_2) = (np.nan, np.nan)
    
    if len(maxima) > 2:
        smr_maxima = x[maxima]
        hsc_maxima = y[maxima]
        grad_mx, itcp_mx = linear_regression(smr_maxima, hsc_maxima)
        eol_3 = (hs_limit - itcp_mx)/grad_mx
        (eol_3, rul_3) = rul_correction(eol_3, max_smr)
    else:
        (eol_3, rul_3) = (np.nan, np.nan)
            
    rul = np.array([rul_1, rul_2, rul_3])
    eol = np.array([eol_1, eol_2, eol_3])
    rul.sort()
    eol.sort()
    
    insight = {
        "gradient_per1000": grad_ex*1000, 
        "intercept": itcp_ex,
        "latest_hs": y[np.where(x==max_smr)][0], 
        "std": np.std(y-make_smooth(y,5)), 
        "earliest_smr": min_smr,
        "latest_smr": max_smr, 
        "rul_optimistic": rul[1] if max_smr <= midlife_smr and np.isnan(rul[1])==False else rul[2],
        "rul_pessimistic": rul[0] if max_smr <= midlife_smr and np.isnan(rul[0])==False else rul[1],
    }
    # additionally compute eol by substracting rul with latest_smr
    ## compute eol_optimistic if rul is not NaN else set value as NaN
    insight["eol_optimistic"] = \
        insight.get("rul_optimistic") - insight.get("latest_smr") \
            if np.isnan(insight.get("rul_optimistic"))==False else np.nan
    ## compute eol_pessimistic if rul is not NaN else set value as NaN
    insight["eol_pessimistic"] = \
        insight.get("rul_pessimistic") - insight.get("latest_smr") \
            if np.isnan(insight.get("rul_pessimistic"))==False else np.nan
    return insight

def linear_regression(x, y):
    from sklearn.linear_model import LinearRegression
    x = np.array(x)
    y = np.array(y)
    linreg = LinearRegression().fit(x.reshape(-1, 1), y)
    score = linreg.score(x.reshape(-1,1), y)
    linear_estimate = linreg.predict(x.reshape(-1, 1))
    gradient = linreg.coef_[0]
    intercept = linreg.intercept_
    return (gradient, intercept)

def locate_extreme(data, window_size):
    maxima_index = []
    minima_index = []
    n = len(data)
    for i in range(n-window_size):
        sub_data = data[i:i+window_size]
        sign_change = np.diff(sub_data)/np.abs(np.diff(sub_data))
        sign_change_diff = np.diff(sign_change)
        max_val = np.max(sub_data)
        min_val = np.min(sub_data)
        if (np.abs(sub_data[0] - sub_data[int(window_size/2)]) >= 0.1) and \
            (np.abs(sub_data[-1] - sub_data[int(window_size/2)]) >= 0.1) :
            maxima = np.where(sign_change_diff<0)[0]
            minima = np.where(sign_change_diff>0)[0]
            for mx_idx in list(maxima):
                if np.abs(sub_data[mx_idx+1]-max_val) < 1E-4:
                    maxima_index.append(i+mx_idx+1)
            for mn_idx in list(minima):
                if np.abs(sub_data[mn_idx+1]-min_val) < 1E-4:
                    minima_index.append(i+mn_idx+1)
    maxima_index = np.array(list(set(maxima_index)))
    minima_index = np.array(list(set(minima_index)))
    maxima_index.sort()
    minima_index.sort()
    return (maxima_index, minima_index)

def logistic_function(x, L=1, k=1, m=0):
    return L/(1+np.exp(k*(m-x)))

def reduce_by_key(data, group_key, sort_key, ascending=False):
    data_dedup = data.sort_values(group_key + sort_key, ascending=ascending)\
        .drop_duplicates(group_key, keep='first').copy()
    for key in group_key:
        data_dedup = data_dedup[data_dedup[key]==data_dedup[key]]
    return data_dedup

def label_vhms_from_fault(vhms_trend, vhms_fault, fault_code, minimum_duration):
    vhms_fault_specific = vhms_fault[(vhms_fault['CODE']==fault_code) & 
                                   (vhms_fault['TOTAL_DURATION_MINUTES'] >= minimum_duration)].copy()
    vhms_fault_specific['NUM_DAYS'] = \
        (vhms_fault_specific['TO_DATE'].astype('datetime64[ns]') - \
             vhms_fault_specific['FROM_DATE'].astype('datetime64[ns]'))/np.timedelta64(1, 'D')
    vhms_fault_specific.columns = ['UNIT_SRL_NUM'] + vhms_fault_specific.columns.tolist()[1:]
    
    pysql = lambda q: sqldf(q, locals())
    
    query = """
        SELECT 
            trend.*
        FROM vhms_fault_specific fault
        INNER JOIN vhms_trend trend
            ON fault.UNIT_SRL_NUM = trend.UNIT_SRL_NUM AND
               trend.TIMESTAMP >= date(fault.FROM_DATE,'-1 day') AND --fault.FROM_DATE - INTERVAL'2 DAYS'
               trend.TIMESTAMP < fault.TO_DATE
        WHERE fault.TOTAL_DURATION_MINUTES/fault.NUM_DAYS >= {0}
        ORDER BY UNIT_SRL_NUM, TIMESTAMP
    """.format(minimum_duration)
    vhms_from_fault = sqldf(query, locals())
    return vhms_from_fault

def join_vhms_with_pap(vhms, pap, time_window=30):
    query = """
        SELECT 
            vhms.*, pap.*
        FROM vhms
        LEFT JOIN pap
            ON pap.SRL_NUM = vhms.UNIT_SRL_NUM AND
               pap.MODL_NUM = vhms.UNIT_MODL AND
               vhms.TIMESTAMP >= pap.SAMPL_DT AND
               vhms.TIMESTAMP < date(pap.SAMPL_DT,'+{0} day')
        ORDER BY UNIT_SRL_NUM, TIMESTAMP, SAMPL_DT DESC
    """.format(time_window)
    
    vhms_x_pap_dataset = reduce_by_key(
        sqldf(query, locals()),
        group_key = ['UNIT_SRL_NUM', 'TIMESTAMP'],
        sort_key = ['SAMPL_DT'])
    
    return vhms_x_pap_dataset

def read_from_file(filename):
    f = open(filename, 'r')
    content = ""
    for row in f:
        if row[0]!='#' and len(row) > 0:
            content += row
    return [c for c in content.split("\n") if len(c) > 0]

def mark_timestamp():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

def make_smooth(data, window_size, iter=1):
    data = np.array(data)
    def moving_average(y, window_size):
        ys = np.copy(y)
        half_window = int(window_size-1/2)
        for i in range(len(y)):
            if i>=half_window:
                ys[i] = np.mean(y[i-half_window:i+half_window])
            else:
                ys[i] = np.mean(y[0:i+1])
        return ys
    
    for i in range(iter):
        data = moving_average(data, window_size)
    return data

def date_add(date, diff):
    if len(date)==10:
        dt_format = '%Y-%m-%d'
    elif len(date)==19:
        dt_format = '%Y-%m-%d %H:%M:%S'
    dt = datetime.strptime(date, dt_format) + timedelta(days=diff)
    return datetime.strftime(dt, dt_format)
    
def get_match_pap(pap_dataset, srl_num, date, component, time_window=30):
    pap = pap_dataset[(pap_dataset['SRL_NUM']==str(srl_num)) & 
                      (pap_dataset['COMPONENT']==component) &
                      (pap_dataset['SAMPL_DT']<=date_add(date, 1)) &
                      (pap_dataset['SAMPL_DT']>date_add(date, -1*time_window))].reset_index(drop=True)
    return pap

def stringify_dict(data):
    for k in data.keys():
        data[k] = str(data.get(k))
    return data

def write_data_to_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
        
def read_data_from_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
        f.close()
    return data
        
def from_pandas_to_json(dataframe):
    data = []
    for idx in dataframe.index:
        datum = dict(dataframe.loc[idx])
        for k in datum.keys():
            datum[k] = str(datum.get(k))
        data.append(datum) 
    return data
    
def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax