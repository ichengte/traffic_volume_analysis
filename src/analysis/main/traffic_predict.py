"""
STL预测流量
"""
import copy
import datetime, os
import logging

import pandas as pd
import numpy as np
import sys
import src.files as files
import src.utils as tools
from src.utils import dt
import matplotlib.colors as mcolors
from adtk.transformer import ClassicSeasonalDecomposition
from tqdm import tqdm
from datetime import datetime, timedelta
from adtk.data import validate_series

rootpath = r'/Users/jct/PycharmProjects/traffic_volume_analysis/'
syspath = sys.path
sys.path = []
sys.path.append(rootpath)  # 将工程根目录加入到python搜索路径中
sys.path.extend([rootpath + i for i in os.listdir(rootpath) if i[0] != '.'])  # 将工程目录下的一级目录添加到python搜索路径中
sys.path.extend(syspath)

colors = list(mcolors.TABLEAU_COLORS.keys())
seasonal = 24
pm_data_dir = files.pm_dir
TRAFFIC_KPI_LIST = ['LTE_DL Traffic Volume(GB)', 'Average TA', 'RRC_Setup_Att_Times', 'ERAB_Setup_Att_Times',
                    'S1Sig Setup Att Times', 'HI Succ Times', 'RRC Conn Users Avg', 'LTE_DL Traffic Volume(GB)']

logging.basicConfig(filename='LOG/STL.log', format='%(filename)s-%(levelname)s:%(message)s', level=logging.DEBUG,
                    filemode='w')


def get_traffic_analysis_result(df, site_cell, normal_dur=None, normal_dur_uncertain=None):
    """
    STL流量预测
    normal_dur, normal_dur_uncertain, 正常期的起始，结束时间，只填写一个，后者可以接受非datetime object
    """
    if (normal_dur is None and normal_dur_uncertain is None) or (
            normal_dur is not None and normal_dur_uncertain is not None):
        raise ValueError('only one of (normal_dur, normal_dur_uncertain) need to be filled.')

    if normal_dur is not None:
        start = normal_dur[0]
        end = normal_dur[1]
    else:
        start, end = normal_dur_uncertain[0], normal_dur_uncertain[1]
        if start == 'StartTime':
            start = df.index[0].to_pydatetime()
        else:
            start = dt(start)
        if end == 'EndTime':
            end = df.index[-1].to_pydatetime()
        else:
            end = dt(end)

    # preprocess
    for col in TRAFFIC_KPI_LIST:
        if all(np.isnan(df[col].values)) is True:
            df = df.drop(col, axis='columns')
            logging.warning('Drop column - {} in {}, all values is Nan.'.format(col, site_cell))

    if len(df.columns) == 0:
        # 如果整个文件都是空的，返回空的异常事件
        return pd.DataFrame(
            columns=['Occurred On (NT)', 'Cleared On (NT)', 'Method', 'Reason', 'Name', 'eNodeB ID', 'Level',
                     'LocalCellRelatedInfo'])
    cols = []

    for c in TRAFFIC_KPI_LIST:
        cols.append("Actual-" + c)
        cols.append("Predict-" + c)
        cols.append("Change-" + c)

    # training data, without anomaly
    train_df = copy.deepcopy(df[(pd.to_datetime(df.index) <= end) & (pd.to_datetime(df.index) >= start)])
    result_df = pd.DataFrame(
        columns=cols)

    if len(train_df) == 0:
        # 如果在目标时间内没有可用于训练的正常数据
        logging.error(
            'No available data during {} and {}, for training the model, cell: {}'.format(normal_dur_uncertain[0],
                                                                                          normal_dur_uncertain[1],
                                                                                          site_cell))
        print('No available data during {} and {}, for training the model, cell: {}'.format(normal_dur_uncertain[0],
                                                                                            normal_dur_uncertain[1],
                                                                                            site_cell))
        return result_df

    for col in TRAFFIC_KPI_LIST:
        if col not in df.columns.tolist():
            return result_df

    train_df.fillna(value=train_df.mean())
    train_df = train_df.loc[:, TRAFFIC_KPI_LIST]
    test_df = df.loc[:, TRAFFIC_KPI_LIST]

    # predict
    for kpi in train_df.columns.tolist():
        train_data = train_df[kpi]
        predictor = ClassicSeasonalDecomposition(freq=seasonal)
        predictor.fit(train_data)

        test_data = validate_series(test_df[kpi])
        test_res = predictor.transform(test_data)
        test_res = test_res.values
        test_array = test_data.values
        pred = test_array - test_res  # 预测的结果
        result_df["Actual-" + kpi] = test_data
        result_df["Predict-" + kpi] = pred
        result_df["Change-" + kpi] = test_array - pred

    for da, row in result_df.iterrows():
        if pd.isna(row["Predict-" + TRAFFIC_KPI_LIST[0]]):
            t = row["Predict-" + TRAFFIC_KPI_LIST[0]]
            while pd.isna(t):
                nd = da + timedelta(days=-1)
                t = result_df.loc[nd, "Predict-" + TRAFFIC_KPI_LIST[0]]
            result_df.loc[da, "Predict-" + TRAFFIC_KPI_LIST[0]] = t
        if pd.isna(row['Actual-' + TRAFFIC_KPI_LIST[0]]):
            result_df.loc[da, 'Change-' + TRAFFIC_KPI_LIST[0]] = -result_df.loc[
                da, 'Predict-' + TRAFFIC_KPI_LIST[0]]

    logging.info('\n')
    return result_df


def analyize(case_idx):
    # case_idx = 3
    case_config_file = 'src/case_config.json'
    # ---------------------------
    case_idx -= 1
    case_name, site_list, alarm_cell_list, neighbor_cell_list, interest_time, normal_duration, label_file_list = tools.load_case_config(
        case_idx, case_config_file)

    save_dir = 'stored_data_{}'.format(case_name)
    tools.check_path(save_dir)

    interest_time[0] = datetime.strptime(interest_time[0], '%Y-%m-%d %H:%M:%S')
    interest_time[1] = datetime.strptime(interest_time[1], '%Y-%m-%d %H:%M:%S')
    s = interest_time[0] + timedelta(hours=1)
    e = interest_time[1] + timedelta(hours=1)

    # read FM data  读取告警文件
    fm_df = pd.read_csv(files.fm_path)
    # get uncleared alarm 在感兴趣时间没有清除的告警
    uncleared_alarms = fm_df[(
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) >= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) < s)) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) <= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) >= interest_time[0])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) >= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) <= interest_time[1])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) <= interest_time[1]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) >= interest_time[1])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) >= interest_time[1]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) < e))
                             ) &
                             (fm_df['eNodeB ID'].isin(site_list))
                             ]
    uncleared_alarms = uncleared_alarms.sort_values(by='Occurred On (NT)', ascending=True)  # 按发生时间的排序
    uncleared_alarms.reset_index(inplace=True, drop=True)
    # print(uncleared_alarms, '\n\n')
    print('case {}'.format(case_idx + 1))
    all_cell_list = alarm_cell_list + neighbor_cell_list

    file_list = os.listdir(pm_data_dir)
    file_list.sort()

    with tqdm(total=len(all_cell_list)) as pbar:
        for cell in all_cell_list:
            pm_file = cell.replace('-', '_') + ".csv"
            if pm_file not in file_list:
                pbar.update()
                continue
            pm_path = os.path.join(pm_data_dir, pm_file)
            pm_df = tools.read_pm_data(pm_path)
            site_cell = pm_file.split('.')[0]

            # predict
            result_df = get_traffic_analysis_result(pm_df, site_cell, normal_dur_uncertain=normal_duration)

            # save
            tools.check_path('{}/traffic'.format(save_dir))
            tools.check_path('{}/traffic/loss/data/neighbor_cell'.format(save_dir))
            tools.check_path('{}/traffic/loss/data/alarm_cell'.format(save_dir))
            if cell in alarm_cell_list:
                result_df.to_csv('{}/traffic/loss/data/alarm_cell/{}.csv'.format(save_dir, cell))
            else:
                result_df.to_csv('{}/traffic/loss/data/neighbor_cell/{}.csv'.format(save_dir, cell))
            pbar.update()


if __name__ == '__main__':
    # case_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    case_list = [7]
    for case_idx in case_list:
        analyize(case_idx)
