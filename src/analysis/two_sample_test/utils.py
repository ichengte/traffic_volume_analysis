import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf

def computeRightK(timeseries, default_lag=24, default_k=5):
    """ compute the right 'k'(subseries length) for timeseries by autocorrelation function

    :param timeseries: pandas series object
    :param default_lag: number of lags to return for acf
    :param default_k: the default value of 'k' when the first peak cannot be found

    :return 'k' value of the timeseries
    """
    acf_lag = acf(timeseries.values, nlags=default_lag)
    for i in range(acf_lag.shape[0]):
        if i > 0 and acf_lag[i] >= acf_lag[i-1] and i < acf_lag.shape[0] and acf_lag[i] >= acf_lag[i+1]:
            return i + 1
    return default_k

def plotTS(kpi_name, timeseries, event_occur_indexes, output_dir):
    """
    :param kpi_name: the name of a certain KPI
    :param timeseres: pandas.series object, correspond to the kpi_name
    :event_occur_indexes: indexes when event happens(dash line)
    """
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    fig_name = os.path.join(output_dir, '{}.png'.format(kpi_name))
    
    plt.title(kpi_name)
    plt.plot(timeseries)
    # plt.legend(loc='upper left')
    # NOTE: event_occur_indexes是调整参数的对应时间点, 这里对应timeseries的下标(从0开始)
    for event_occur_idx in event_occur_indexes:
        plt.axvline(x=event_occur_idx, color='grey', linestyle='--')
    plt.savefig(fig_name)
    plt.close()

def getEventOccurIndexes(kpi_df, occur_timestamps):
    last_timestamp = kpi_df.loc[0]['Time']
    occur_time_idx = 0
    occur_indexes = []
    for i, row in kpi_df.iterrows():
        cur_timestamp = row['Time']
        if i > 0 and occur_time_idx < len(occur_timestamps) and \
            last_timestamp < occur_timestamps[occur_time_idx] and cur_timestamp >= occur_timestamps[occur_time_idx]:
            occur_indexes.append(i)
            occur_time_idx += 1      
        if occur_time_idx >= len(occur_timestamps):
            break
        last_timestamp = cur_timestamp

    return occur_indexes