import datetime, os, pickle, json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal

alarm_sep = '*__*'
kg_sep = '/'


def dt(x):
    dt_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return dt_obj


def check_path(p):
    if os.path.exists(p) is False: os.makedirs(p)


def save_to_pickle(data, pkl_file):
    f = open(pkl_file, mode='wb')
    pickle.dump(data, f)
    f.close()


def read_from_pickle(pkl_file):
    f = open(pkl_file, mode='rb')
    data = pickle.load(f)
    f.close()
    return data


def read_fm_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Occurred On (NT)', 'Cleared On (NT)'])
    return df


def read_pm_data(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col='Start Time')
    return df


def convert_str_to_timestamp(x):
    d_bj = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    d_delta = datetime.timedelta(days=0, hours=8)  # BJ = UTC + 8
    d_utc = d_bj - d_delta  # UTC = BJ - 8
    timestamp = (d_utc - datetime.datetime(1970, 1, 1)).total_seconds()
    return timestamp


def convert_timestamp_to_str(x):
    d = datetime.datetime.utcfromtimestamp(x)  # UTC时间
    d_delta = datetime.timedelta(days=0, hours=8)  # BJ = UTC + 8
    d_cest = d + d_delta
    time_string = d_cest.strftime('%Y-%m-%d %H:%M:%S')
    return time_string


def get_case_config(case_idx, case_config_file):
    with open(case_config_file, 'r', encoding='utf-8') as file_job:
        configs = json.load(file_job)
        file_job.close()
    case_config = configs[case_idx]
    normal_duration = case_config['normal_duration']
    info = pd.read_csv(case_config['information_file'])
    alarm_cell_list = []  # 该案例所有涉及的所有告警小区
    for _, row in info.iterrows():
        alarm_cell = row['alarm_cell_list']
        if '|' in alarm_cell:
            alarm_cell = alarm_cell.split('|')
            alarm_cell_list = alarm_cell_list + alarm_cell
        else:
            alarm_cell_list.append(alarm_cell)
    alarm_cell_list = list(set(alarm_cell_list))
    alarm_cell_list = ['{}-{}'.format(cell.split('_')[0], cell.split('_')[1]) for cell in alarm_cell_list]
    alarm_cell_list.sort()
    site_list = [int(cell_id.split('-')[0]) for cell_id in alarm_cell_list]
    site_list = list(set(site_list))
    site_list.sort()
    case_name = case_config['case_name']
    interest_time = case_config['interest_time']
    label_file_list = case_config['label_files']
    return case_name, site_list, alarm_cell_list, interest_time, normal_duration, label_file_list


def load_case_config(case_idx, case_config_file):
    alarm_cell_list = []
    affected_cell_list = []
    with open(case_config_file, 'r', encoding='utf-8') as file_job:
        configs = json.load(file_job)
        file_job.close()
    case_config = configs[case_idx]
    normal_duration = case_config['normal_duration']
    info = pd.read_csv(case_config['information_file'])

    for _, row in info.iterrows():
        alarm_cell = row['alarm_cell_list']
        if '|' in alarm_cell:
            alarm_cell = alarm_cell.split('|')
            alarm_cell_list = alarm_cell_list + alarm_cell
        else:
            alarm_cell_list.append(alarm_cell)
    alarm_cell_list = list(set(alarm_cell_list))
    alarm_cell_list = ['{}-{}'.format(cell.split('_')[0], cell.split('_')[1]) for cell in alarm_cell_list]
    alarm_cell_list.sort()

    for _, row in info.iterrows():
        affected_cell = row['affected_cell_list']
        if '|' in affected_cell:
            affected_cell = affected_cell.split('|')
            affected_cell_list = affected_cell_list + affected_cell
        else:
            affected_cell_list.append(affected_cell)
    affected_cell_list = list(set(affected_cell_list))
    neighbor_cell_list = ['{}-{}'.format(cell.split('_')[0], cell.split('_')[1]) for cell in affected_cell_list if
                          cell.replace('_', '-') not in alarm_cell_list]

    alarm_site_list = [int(cell_id.split('-')[0]) for cell_id in alarm_cell_list]
    alarm_site_list = list(set(alarm_site_list))
    alarm_site_list.sort()
    case_name = case_config['case_name']
    interest_time = case_config['label_files'][0]['time_of_interest'].split('||')
    label_file_list = case_config['label_files']
    return case_name, alarm_site_list, alarm_cell_list, neighbor_cell_list, interest_time, normal_duration, label_file_list


def get_group_list(case_idx):
    case_idx -= 1
    case_config_file = 'src/case_config.json'

    with open(case_config_file, 'r', encoding='utf-8') as file_job:
        configs = json.load(file_job)
        file_job.close()
    case_config = configs[case_idx]
    info = pd.read_csv(case_config['information_file'])

    save_dir = 'stored_data_{}'.format(case_config['case_name'])
    check_path(save_dir)

    alarm_site_list = []
    for _, row in info.iterrows():
        li = row['alarm_cell_list'].split('|')
        alarm_site_list += [cell.split('_')[0] for cell in li]

    group_list = []
    for i, row in info.iterrows():
        alarm_cell_list = row['alarm_cell_list'].split('|')
        neighbor_cell_list = row['affected_cell_list'].split('|')
        neighbor_cell_list = [cell for cell in neighbor_cell_list if cell not in alarm_cell_list]
        tmp_dict = {'alarm_cell_list': alarm_cell_list,
                    'neighbor_cell_list': neighbor_cell_list, 'group_id': row['group_id']}
        group_list.append(tmp_dict)

    return group_list


def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


def get_time_list(df, cell_alarms_list=None, interest_time=None, uncleared_alarms=None, idx=None):
    time_list = []
    if uncleared_alarms is None:
        for alarm in cell_alarms_list:
            start = alarm["Occurred On (NT)"][:-5] + "00:00"
            start = pd.date_range(start, freq='H', periods=2).strftime('%Y-%m-%d %H:%M:%S').values[1]
            end = interest_time
            if pd.to_datetime(alarm["Cleared On (NT)"]) > pd.to_datetime(df.index[-1]):
                end = df.index[-1]
            time_list += pd.date_range(start, end, freq='H')
    else:
        for _, alarm in uncleared_alarms.iterrows():
            start = alarm["Occurred On (NT)"][:-5] + "00:00"
            end = alarm["Cleared On (NT)"]
            if pd.to_datetime(alarm["Cleared On (NT)"]) > pd.to_datetime(df.index[-1]):
                end = df.index[-1]
            end = str(end)
            if end[-5:] == "00:00":
                end = pd.date_range(end, freq='H', periods=2).strftime('%Y-%m-%d %H:%M:%S').values[1]
            time_list += pd.date_range(start, end, freq='H')

    time_list = list(set(time_list))
    if idx is not None:
        time_list = time_list[0:idx]
    return sorted(time_list)


class CellHandler:
    def __init__(self, ep_path, bbu_path, rru_path):
        self.ep_data_file = ep_path
        self.bbu_data_file = bbu_path
        self.rru_data_file = rru_path

        self.bbu_data = pd.read_csv(self.bbu_data_file)
        self.rru_data = pd.read_csv(self.rru_data_file)
        self.ep_data = pd.read_csv(self.ep_data_file)

    def get_Cellid_By_BBU(self, site_id, bbu_id):
        cell_id_list = self.bbu_data[(pd.to_numeric(self.bbu_data['eNodeBId'], errors='ignore') == int(site_id)) & (
                    self.bbu_data['BBUinfo'] == bbu_id)]['LocalCell Id'].values.tolist()
        return cell_id_list

    def get_Cellid_By_RRU(self, site_id, rru_id):
        cell_id_list = self.rru_data[(pd.to_numeric(self.rru_data['eNodeBId'], errors='ignore') == int(site_id)) & (
                    self.rru_data['RRUinfo'] == rru_id)]['LocalCell Id'].values.tolist()
        return cell_id_list

    def get_Cellid_By_Site(self, site_id):
        cell_id_list = self.ep_data[pd.to_numeric(self.ep_data['eNodeBID']) == int(site_id)][
            'LocalCellID'].values.tolist()
        return cell_id_list

    def get_Cell_Neighbors(self, site_cell_str):
        inter_ = self.ep_data[self.ep_data['eNodeBidLocalCellID'] == site_cell_str]
        if len(inter_) > 0:
            inter = inter_.iloc[0]['InterNeighbor']
        else:
            inter = inter_

        intra_ = self.ep_data[self.ep_data['eNodeBidLocalCellID'] == site_cell_str]
        if len(intra_) > 0:
            intra = intra_.iloc[0]['IntraNeighbor']
        else:
            intra = intra_

        # 不存在这个小区在EP中
        if len(inter_) == 0 and len(intra_) == 0:
            return []
        else:
            # 查询到了，但没有邻区记录
            if isinstance(inter, str) is False:
                inter = []
            elif ';' in inter:
                inter = inter.split(';')
            else:
                inter = [inter]

            if isinstance(intra, str) is False:
                intra = []
            elif ';' in intra:
                intra = intra.split(';')
            else:
                intra = [intra]
            nei_list = inter + inter
            nei_list = list(set(nei_list))
            return nei_list
