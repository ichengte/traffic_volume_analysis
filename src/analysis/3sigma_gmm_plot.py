"""
根据告警前的正常数据采用gmm建模，计算toi时间段的流量损失
"""
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from src.analysis.main.traffic_predict import TRAFFIC_KPI_LIST

from src import files
import src.utils as tools
from src.utils import get_group_list, load_case_config, Gaussian_Distribution

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS.keys())


def dt(x):
    dt_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return dt_obj


def three_sigma_max(traffic_ser, th, mu, var, maximum=True):
    """
    默认求异常大流量
    maximum=False求异常小流量
    """
    if maximum:
        tmp_ser = traffic_ser[(mu + th * var) < traffic_ser]
        data = tmp_ser.values.tolist()
        t = sum(data) - len(data) * mu
    else:
        tmp_ser = traffic_ser[(mu - th * var) > traffic_ser]
        data = tmp_ser.values.tolist()
        t = sum(data) - len(data) * mu
    return round(t, 2)


def get_clean_traffic(ser, th):
    pass


def get_best_comp(x_train, y_true):
    n_comp = 1
    loss = 99999999999
    last_m_list, last_var_list, last_wei_list = [], [], []
    for i in range(1, 4):
        gmm = GaussianMixture(n_components=i).fit(x_train)
        m_list = gmm.means_
        var_list = gmm.covariances_
        wei_list = gmm.weights_

        y_pred = np.array([0] * len(y_true), dtype=np.float64)
        for (m, var, w) in zip(m_list, var_list, wei_list):
            _, gaussian = Gaussian_Distribution(N=1, m=m, sigma=var.squeeze())
            y_pred += np.array(gaussian.pdf(y_true) * w)
        cur_loss = mean_squared_error(y_pred, y_true)
        if cur_loss < loss:
            loss = cur_loss
            n_comp = i
            last_m_list, last_var_list, last_wei_list = m_list, var_list, wei_list
    # print(loss)
    return n_comp, last_m_list, last_var_list, last_wei_list


def main2(case_idx):
    pm_data_dir = files.pm_dir
    case_config_file = 'src/case_config.json'
    # ---------------------------
    case_idx -= 1
    case_name, alarm_site_list, alarm_cell_list, neighbor_cell_list, interest_time, normal_duration, label_file_list = load_case_config(
        case_idx, case_config_file)
    interest_time[0] = datetime.strptime(interest_time[0], '%Y-%m-%d %H:%M:%S')
    interest_time[1] = datetime.strptime(interest_time[1], '%Y-%m-%d %H:%M:%S')
    s = interest_time[0] + timedelta(hours=1)
    e = interest_time[1] + timedelta(hours=1)

    toi_list = pd.date_range(interest_time[0], interest_time[1], freq='H')

    save_dir = 'stored_data_{}'.format(case_name)
    tools.check_path(save_dir)
    alarm_cell_csv_path = '{}/traffic/loss/data/alarm_cell'.format(save_dir)
    neighbor_cell_csv_path = '{}/traffic/loss/data/neighbor_cell'.format(save_dir)

    fm_df = pd.read_csv(files.fm_path)
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
    group_list = get_group_list(case_idx + 1)

    total = 0
    for group in group_list:
        alarm_cell_list = group['alarm_cell_list']
        neighbor_cell_list = group['neighbor_cell_list']
        total += len(alarm_cell_list) + len(neighbor_cell_list)
    total *= 2

    file_list = os.listdir(pm_data_dir)
    file_list.sort()

    print('Case:{}'.format(case_idx + 1))
    for group in group_list:
        tmp_cell_list = []  # cell
        result_list = [[], [], [], []]  # 3sigma and stl
        tmp_type_list = []
        alarm_cell_list = group['alarm_cell_list']
        neighbor_cell_list = group['neighbor_cell_list']
        cell_list = alarm_cell_list + neighbor_cell_list

        stl_df = pd.read_csv('{}/traffic/loss/result/case{}.csv'.format(save_dir, case_idx + 1))
        stl_df.set_index('Cell', inplace=True)

        for i, cell in enumerate(cell_list):
            pm_file = cell.replace('-', '_') + ".csv"
            if pm_file not in file_list:
                # pbar.update(2)
                continue
            if cell.replace('_', '-') not in stl_df.index:
                continue
            result_list[3].append(float(stl_df.loc[cell.replace('_', '-'), TRAFFIC_KPI_LIST[0]]))
            tmp_type_list.append(stl_df.loc[cell.replace('_', '-'), 'Type'])

            if cell in alarm_cell_list:
                df = pd.read_csv(os.path.join(alarm_cell_csv_path, cell.replace('_', '-') + '.csv'))
            else:
                df = pd.read_csv(os.path.join(neighbor_cell_csv_path, cell.replace('_', '-') + '.csv'))

            # handle start time and end time
            df.set_index('Start Time', inplace=True)
            start = normal_duration[0]
            end = normal_duration[1]
            if start == 'StartTime':
                start = dt(df.index[0])
            else:
                start = dt(start)
            if end == 'EndTime':
                end = dt(df.index[-1])
            else:
                end = dt(end)

            # normal data
            df_normal = df[(pd.to_datetime(df.index) <= end) & (pd.to_datetime(df.index) >= start)]
            s_normal = df_normal['Change-' + TRAFFIC_KPI_LIST[0]].astype(np.float).values.tolist()

            # alarm data
            df.index = pd.to_datetime(df.index)
            # time_list = get_time_list(df, uncleared_alarms=uncleared_alarms)
            df_alarm = df[df.index.isin(toi_list)]
            s_alarm = df_alarm['Change-' + TRAFFIC_KPI_LIST[0]].astype(np.float).values.tolist()

            # plot
            s_list = [s_normal, s_alarm]
            for j, s in enumerate(s_list):
                plt.figure()
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                bins_value_list, bins, _ = ax1.hist(s, density=True)

                # gaussian
                if j == 0:
                    # three sigma normal
                    x_train = np.expand_dims(s_normal, 1)
                    n_comp, mu_list, var_list, wei_list = get_best_comp(x_train, s_normal)

                    # x data
                    x = np.linspace(min(s) - 0.5, max(s) + 0.5, 5000)

                    # bin_value max
                    y_max1 = max(bins_value_list)
                    y_max2 = 0
                    cnt = 1
                    for (m, var, w) in zip(mu_list, var_list, wei_list):
                        _, gaussian = Gaussian_Distribution(N=1, m=m.squeeze(), sigma=var.squeeze())
                        y = gaussian.pdf(x)
                        ax2.plot(x, y, colors[cnt + 1],
                                 label=r'$\mu_{} ={:.2f},\sigma_{} ={:.2f}$'.format(cnt, m.squeeze(), cnt,
                                                                                    var.squeeze()))
                        y_max2 = max(y_max2, max(y))
                        cnt += 1

                    # max mu and var
                    mu_list = np.array(mu_list).reshape(-1).tolist()
                    var_list = np.array(var_list).reshape(-1).tolist()
                    if cell not in alarm_cell_list:
                        mu = max(mu_list)
                        var = var_list[mu_list.index(mu)]
                    else:
                        mu = min(mu_list)
                        var = var_list[mu_list.index(mu)]
                else:
                    # 3sigma alarm cell，plot mu - 3sigma
                    if cell in alarm_cell_list:
                        t1 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 1, mu, var,
                                             maximum=False)
                        t2 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 2, mu, var,
                                             maximum=False)
                        t3 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 3, mu, var,
                                             maximum=False)
                        ax1.axvline(mu - 3 * var, c='r', ls="--")
                        ax1.text(mu - 3 * var,
                                 y_max1 * 0.5,
                                 r'$\mu-3\sigma={}$'.format(round(mu - 3 * var, 3)),
                                 c='r',
                                 ha='center')

                        ax1.axvline(mu - 2 * var, c='g', ls="--")
                        ax1.text(mu - 2 * var,
                                 y_max1 * 0.3,
                                 r'$\mu-2\sigma={}$'.format(round(mu - 2 * var, 3)),
                                 c='g',
                                 ha='center')

                        ax1.axvline(mu - 1 * var, c='b', ls="--")
                        ax1.text(mu - 1 * var,
                                 y_max1 * 0.1,
                                 r'$\mu-\sigma={}$'.format(round(mu - 1 * var, 3)),
                                 c='b',
                                 ha='center')
                    else:
                        # 3sigma neighbor cell，plot mu + 3sigma
                        t1 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 1, mu, var)
                        t2 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 2, mu, var)
                        t3 = three_sigma_max(df_alarm['Change-' + TRAFFIC_KPI_LIST[0]], 3, mu, var)
                        ax1.axvline(mu + 3 * var, c='r', ls="--")
                        ax1.text(mu + 3 * var,
                                 y_max1 * 0.5,
                                 r'$\mu+3\sigma={}$'.format(round(mu + 3 * var, 3)),
                                 c='r',
                                 ha='center')

                        ax1.axvline(mu + 2 * var, c='g', ls="--")
                        ax1.text(mu + 2 * var,
                                 y_max1 * 0.3,
                                 r'$\mu+2\sigma={}$'.format(round(mu + 2 * var, 3)),
                                 c='g',
                                 ha='center')

                        ax1.axvline(mu + 1 * var, c='b', ls="--")
                        ax1.text(mu + 1 * var,
                                 y_max1 * 0.1,
                                 r'$\mu+\sigma={}$'.format(round(mu + 1 * var, 3)),
                                 c='b',
                                 ha='center')

                    # df columns: cell_list
                    tmp_cell_list.append(cell)
                    result_list[0].append(t1)
                    result_list[1].append(t2)
                    result_list[2].append(t3)

                ax2.set_ylim(0, y_max2)

                # plot info
                ax1.set_xlabel('loss')
                ax1.set_ylabel(r'hist probability density')
                ax2.set_ylabel(r'gaussian probability density')
                if cell in alarm_cell_list:
                    plt.title('alarm cell:{}'.format(cell))
                else:
                    plt.title('neighbor cell:{}'.format(cell))
                ax1.grid(True)
                if j == 1:
                    tools.check_path(
                        'stored_data_{}/traffic/3sigma_extend/during alarm/{}'.format(case_name,
                                                                                      group['group_id']))
                    # plt.legend(loc=2)
                    plt.savefig(
                        'stored_data_{}/traffic/3sigma_extend/during alarm/{}/{}.png'.format(case_name,
                                                                                             group[
                                                                                                 'group_id'],
                                                                                             cell.replace(
                                                                                                 '_',
                                                                                                 '-')))
                else:
                    tools.check_path(
                        'stored_data_{}/traffic/3sigma_extend/before alarm/{}'.format(case_name,
                                                                                      group['group_id']))
                    plt.legend(loc=2)
                    plt.savefig(
                        'stored_data_{}/traffic/3sigma_extend/before alarm/{}/{}.png'.format(case_name,
                                                                                             group[
                                                                                                 'group_id'],
                                                                                             cell.replace(
                                                                                                 '_',
                                                                                                 '-')))
                # pbar.update()
                plt.close()
        tmp_df = pd.DataFrame()
        tmp_cell_list = [cell.replace('_', '-') for cell in tmp_cell_list]

        tmp_df['Cell'] = tmp_cell_list
        tmp_df['Type'] = tmp_type_list
        tmp_df['1sigma'] = result_list[0]
        tmp_df['2sigma'] = result_list[1]
        tmp_df['3sigma'] = result_list[2]
        tmp_df['stl'] = result_list[3]

        # neighbor ascend
        k = len(alarm_cell_list)
        tmp_df.loc[len(tmp_df), '1sigma'] = round(sum(result_list[0][k:]), 2)
        tmp_df.loc[len(tmp_df) - 1, '2sigma'] = round(sum(result_list[1][k:]), 2)
        tmp_df.loc[len(tmp_df) - 1, '3sigma'] = round(sum(result_list[2][k:]), 2)
        tmp_df.loc[len(tmp_df) - 1, 'stl'] = round(sum(result_list[3][k:]), 2)
        tmp_df.loc[len(tmp_df) - 1, 'Cell'] = '邻区增加'

        # alarm descend
        tmp_df.loc[len(tmp_df), '1sigma'] = round(sum(result_list[0][:k]), 2)
        tmp_df.loc[len(tmp_df) - 1, '2sigma'] = round(sum(result_list[1][:k]), 2)
        tmp_df.loc[len(tmp_df) - 1, '3sigma'] = round(sum(result_list[2][:k]), 2)
        tmp_df.loc[len(tmp_df) - 1, 'stl'] = round(sum(result_list[3][:k]), 2)
        tmp_df.loc[len(tmp_df) - 1, 'Cell'] = '告警小区减少'

        # total
        tmp_df.loc[len(tmp_df), '1sigma'] = round(
            tmp_df.loc[len(tmp_df) - 2, '1sigma'] + tmp_df.loc[len(tmp_df) - 3, '1sigma'], 2)
        tmp_df.loc[len(tmp_df) - 1, '2sigma'] = round(
            tmp_df.loc[len(tmp_df) - 2, '2sigma'] + tmp_df.loc[len(tmp_df) - 3, '2sigma'], 2)
        tmp_df.loc[len(tmp_df) - 1, '3sigma'] = round(
            tmp_df.loc[len(tmp_df) - 2, '3sigma'] + tmp_df.loc[len(tmp_df) - 3, '3sigma'], 2)
        tmp_df.loc[len(tmp_df) - 1, 'stl'] = round(
            tmp_df.loc[len(tmp_df) - 2, 'stl'] + tmp_df.loc[len(tmp_df) - 3, 'stl'], 2)
        tmp_df.loc[len(tmp_df) - 1, 'Cell'] = '总流量'

        # save
        tools.check_path('stored_data_{}/traffic/3sigma_extend/result_by_group'.format(case_name))
        tmp_df.to_csv('stored_data_{}/traffic/3sigma_extend/result_by_group/{}.csv'.format(case_name,
                                                                                           group[
                                                                                               'group_id']),
                      index=False, encoding='utf_8_sig')
        a = tmp_df.loc[len(tmp_df) - 1, '3sigma']
        b = tmp_df.loc[len(tmp_df) - 2, '3sigma']
        c = tmp_df.loc[len(tmp_df) - 3, '3sigma']
        print('\n{}\n告警小区减少：{}，邻区增加：{}，总流量：{}\n'.format(group['group_id'], b, c, a))


if __name__ == '__main__':
    # case_list = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    case_list = [4, 13]
    for case_idx in case_list:
        main2(case_idx)
