"""
计算toi时间段的STL的流量损失
"""

import logging
import os.path
from datetime import datetime, timedelta

import pandas as pd
import src.utils as tools
from src.utils import load_case_config
from src import files
from src.analysis.main.traffic_predict import TRAFFIC_KPI_LIST
from src.utils import CellHandler

pm_data_dir = files.pm_dir


def get_alarm_cell_related_dict(uncleared_alarms):
    """
    根据拓扑关系，将告警关联到具体的小区
    """
    # 通过感兴趣的告警将告警类型关联具体小区
    handler = CellHandler(ep_path=files.ep_path, bbu_path=files.bbu_path, rru_path=files.rru_path)
    alarm_cell_related_dict = {}
    for i, a in uncleared_alarms.iterrows():
        # 该告警影响了哪些小区？
        target_cell_list = []
        if a['Level'] == 'Site':
            cell_id_list = handler.get_Cellid_By_Site(a['eNodeB ID'])
            for cell_id in cell_id_list:
                target_cell_list.append('{}-{}'.format(a['eNodeB ID'], cell_id))
        elif a['Level'] == 'Cell':
            cell_id = a['LocalCellRelatedInfo']
            target_cell_list.append('{}-{}'.format(a['eNodeB ID'], cell_id))
        elif a['Level'] == 'BBU':
            cell_id_list = handler.get_Cellid_By_BBU(a['eNodeB ID'], a['LocalCellRelatedInfo'])
            for cell_id in cell_id_list:
                target_cell_list.append('{}-{}'.format(a['eNodeB ID'], cell_id))
        elif a['Level'] == 'RRU':
            cell_id_list = handler.get_Cellid_By_RRU(a['eNodeB ID'], a['LocalCellRelatedInfo'])
            for cell_id in cell_id_list:
                target_cell_list.append('{}-{}'.format(a['eNodeB ID'], cell_id))

        for cell in target_cell_list:
            if cell in alarm_cell_related_dict.keys():
                alarm_cell_related_dict[cell].append(a)
            else:
                alarm_cell_related_dict[cell] = [a]

    return alarm_cell_related_dict


def compute_data_loss(case_idx):
    case_config_file = 'src/case_config.json'
    case_idx -= 1
    case_name, alarm_site_list, alarm_cell_list, neighbor_cell_list, interest_time, normal_duration, label_file_list = load_case_config(
        case_idx, case_config_file)
    all_cell_list = alarm_cell_list + neighbor_cell_list

    interest_time[0] = datetime.strptime(interest_time[0], '%Y-%m-%d %H:%M:%S')
    interest_time[1] = datetime.strptime(interest_time[1], '%Y-%m-%d %H:%M:%S')
    s = interest_time[0] + timedelta(hours=1)
    e = interest_time[1] + timedelta(hours=1)

    toi_list = pd.date_range(interest_time[0], interest_time[1], freq='H')

    save_dir = 'stored_data_{}'.format(case_name)
    tools.check_path(save_dir)
    alarm_cell_path = '{}/traffic/loss/data/alarm_cell'.format(save_dir)
    neighbor_cell_path = '{}/traffic/loss/data/neighbor_cell'.format(save_dir)

    file_list = os.listdir(pm_data_dir)
    file_list.sort()

    # read fm data
    fm_df = pd.read_csv(files.fm_path)
    # get interest alarm
    uncleared_alarms = fm_df[(
                                     ((pd.to_datetime(fm_df['Cleared On (NT)']) >= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) < s)) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) <= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) >= interest_time[0])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) >= interest_time[0]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) <= interest_time[1])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) <= interest_time[1]) & (
                                             pd.to_datetime(fm_df['Cleared On (NT)']) >= interest_time[1])) |
                                     ((pd.to_datetime(fm_df['Occurred On (NT)']) >= interest_time[1]) & (
                                             pd.to_datetime(fm_df['Occurred On (NT)']) < e))
                             ) &
                             (fm_df['eNodeB ID'].isin(alarm_site_list))
                             ]

    loss_df = pd.DataFrame(columns=["Cell", "Type", 'Name', 'Count'] + TRAFFIC_KPI_LIST)

    cell_list = alarm_cell_list + neighbor_cell_list
    type_list = []
    loss_list = []

    alarm_cell_related_dict = get_alarm_cell_related_dict(uncleared_alarms)

    # compute loss
    for cell in all_cell_list:
        pm_file = cell.replace('-', '_') + ".csv"
        if pm_file not in file_list:
            cell_list.remove(cell)
            continue
        if cell in alarm_cell_list:
            df = pd.read_csv(os.path.join(alarm_cell_path, cell + '.csv'))
        else:
            df = pd.read_csv(os.path.join(neighbor_cell_path, cell + '.csv'))
        if "Start Time" not in df.columns.tolist():
            cell_list.remove(cell)
            logging.warning("Start Time not in df.columns")
            return loss_df

        df1 = df.set_index("Start Time")
        df1["Actual-" + TRAFFIC_KPI_LIST[0]].fillna(value=0, inplace=True)
        if cell in alarm_cell_list:
            type_list.append("alarm")
        else:
            type_list.append("neighbor")

        # add
        s = 0.0
        for time in toi_list:
            time = time.strftime('%Y-%m-%d %H:%M:%S')
            if time not in df1.index:
                continue
            s += float(df1.loc[time, "Change-" + TRAFFIC_KPI_LIST[0]])
        loss_list.append(s)

    loss_df['Cell'] = cell_list
    loss_df['Type'] = type_list
    loss_df[TRAFFIC_KPI_LIST[0]] = loss_list
    for i, row, in loss_df.iterrows():
        loss_df.loc[i, 'LTE_DL Traffic Volume(GB)'] = round(row['LTE_DL Traffic Volume(GB)'], 2)

    for cell in alarm_cell_related_dict.keys():
        k = len(alarm_cell_related_dict[cell])
        t_name_list = [a['Name'] for a in alarm_cell_related_dict[cell]]
        t_name_list = list(set(t_name_list))
        t_name_list.sort()
        alarm_cell_related_dict[cell] = [t_name_list, k]

    loss_df.set_index('Cell', inplace=True)
    for cell in alarm_cell_related_dict.keys():
        # 这里通过拓扑关系来关联告警小区会和info文件中的有冲突：可能在info文件中没有这个告警小区，我们就跳过它
        if cell not in cell_list or cell.find('MPT') != -1:
            continue
        alarm_cell_related_dict[cell][0].sort()
        loss_df.loc[cell, 'Name'] = ','.join(alarm_cell_related_dict[cell][0])
        loss_df.loc[cell, 'Count'] = alarm_cell_related_dict[cell][1]

    result_path = "{}/traffic/loss/result/".format(save_dir)
    tools.check_path(result_path)
    loss_df.fillna(value='None', inplace=True)
    loss_df.sort_values(by=['Type', 'LTE_DL Traffic Volume(GB)', 'Cell'], inplace=True)
    loss_df.to_csv(result_path + "case{}.csv".format(case_idx + 1))


if __name__ == '__main__':
    # case_list = [2, 3, 4, 5, 6,7, 8,9, 10, 11,12, 13, 14, 15, 16, 17, 18]
    case_list = [4, 13]
    for case_idx in case_list:
        compute_data_loss(case_idx)
