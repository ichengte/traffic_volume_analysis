import pandas as pd
import numpy as np

class InfoExtractor(object):
    """ 信息提取类
    
    主用用于从小区对应的kpi_df(指标)和cfg_df(配置)中提取出相关信息

    """

    def __init__(self) -> None:
        pass

    def getParamNames(self, cfg_df):
        # HACK: 配置列名可能出现不一样的情况，这里先粗糙处理，待后续改进(需要注意！！！)
        drop_columns = ['cell_id', 'approx_execution_time']
        column_names = list(cfg_df.columns)
        param_names = []
        for column_name in column_names:
            if column_name not in drop_columns:
                param_names.append(column_name)
        return param_names
    
    def getKPINames(self, kpi_df):
        # HACK: 配置列名可能出现不一样的情况，这里先粗糙处理，待后续改进(需要注意！！！)
        drop_columns = ['cell_id', 'Time']
        column_names = list(kpi_df.columns)
        kpi_names = []
        for column_name in column_names:
            if column_name not in drop_columns:
                kpi_names.append(column_name)
        return kpi_names

    def getEventOccurTimes(self, cfg_df):
        """
        :param cfg_df: all parameters adjustment records

        :return eventOccurDict_perParam: {
            'param1': [timestamp_1, ..., timestamp_m],
            ...
        }
        """
        eventOccurDict_perParam = {}
        param_names = self.getParamNames(cfg_df)
        for param_name in param_names:
            eventOccurDict_perParam[param_name] = []

        last_row = cfg_df.iloc[0, :]
        for _, row in cfg_df.iterrows():
            cur_timestamp = row['approx_execution_time']
            # NOTE: 这里只要某个参数发生了变化，就认为该参数发生了对应的事件
            for param_name in param_names:
                if row[param_name] != last_row[param_name]:
                    eventOccurDict_perParam[param_name].append(cur_timestamp)
            last_row = row
        return eventOccurDict_perParam

    def getRearDataPerParamPerKPI(self, kpi_df, kpi_name, occur_timestamps, subseries_windowsz):
        """
        :param kpi_df: all kpis of a cell
        :param occur_timestamps: timestamp list when configs(certainly one param is changed) are changed
        :param subseries_windowsz: the window size of rear subsereis(G), which is computed by utils.computeRighK

        :return rear_kpi_list: kpi subseries after configs are changed
        """
        last_timestamp = kpi_df.loc[0]['Time']
        occur_time_idx = 0
        rear_kpi_list = []
        for i, row in kpi_df.iterrows():
            cur_timestamp = row['Time']
            if i > 0 and occur_time_idx < len(occur_timestamps) and \
                last_timestamp < occur_timestamps[occur_time_idx] and cur_timestamp >= occur_timestamps[occur_time_idx]:
                rear_kpi_list.append(kpi_df.loc[i: i+subseries_windowsz-1, kpi_name].values.tolist())
                occur_time_idx += 1
            if occur_time_idx >= len(occur_timestamps):
                break
            last_timestamp = cur_timestamp
        
        return rear_kpi_list