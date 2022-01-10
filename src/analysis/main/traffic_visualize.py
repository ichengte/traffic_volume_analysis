"""
可视化流量真实值和预测值
"""
import os.path
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import src.utils as tools
from src import files
from src.utils import load_case_config
import seaborn as sns
from datetime import datetime, timedelta
from traffic_predict import TRAFFIC_KPI_LIST
from src.analysis.main.compute_traffic_loss import get_alarm_cell_related_dict
import matplotlib.patches as patches

def dt(x):
    dt_obj = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return dt_obj


def get_all_alarm_region(uncleared_alarms):
    l_alarm = '2050-11-11 00:00:00'
    r_alarm = '2000-11-11 00:00:00'
    for _, alarm in uncleared_alarms.iterrows():
        if alarm['Occurred On (NT)'] < l_alarm:
            l_alarm = alarm['Occurred On (NT)']

        if alarm['Cleared On (NT)'] > r_alarm:
            r_alarm = alarm['Cleared On (NT)']

    return [l_alarm, r_alarm]


def visualize(case_idx):
    pm_data_dir = files.pm_dir
    case_config_file = 'src/case_config.json'
    # ---------------------------
    case_idx -= 1
    case_name, alarm_site_list, alarm_cell_list, neighbor_cell_list, interest_time, normal_duration, label_file_list = load_case_config(
        case_idx, case_config_file)

    save_dir = 'stored_data_{}'.format(case_name)
    tools.check_path(save_dir)
    alarm_cell_csv_path = '{}/traffic/loss/data/alarm_cell'.format(save_dir)
    neighbor_cell_csv_path = '{}/traffic/loss/data/neighbor_cell'.format(save_dir)

    alarm_cell_fig_path = '{}/traffic/loss/fig/alarm_cell'.format(save_dir)
    neighbor_cell_fig_path = '{}/traffic/loss/fig/neighbor_cell'.format(save_dir)
    tools.check_path(alarm_cell_fig_path)
    tools.check_path(neighbor_cell_fig_path)

    file_list = os.listdir(pm_data_dir)
    file_list.sort()

    # read FM data  读取告警文件
    fm_df = pd.read_csv(files.fm_path)
    # get uncleared alarm 在感兴趣时间没有清除的告警
    interest_time[0] = datetime.strptime(interest_time[0], '%Y-%m-%d %H:%M:%S')
    interest_time[1] = datetime.strptime(interest_time[1], '%Y-%m-%d %H:%M:%S')
    s = interest_time[0] + timedelta(hours=1)
    e = interest_time[1] + timedelta(hours=1)
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
    # time of interest list
    toi_list = pd.date_range(interest_time[0], interest_time[1], freq='H')

    # fig
    loss_df = pd.read_csv("{}/traffic/loss/result/".format(save_dir) + 'case{}.csv'.format(case_idx + 1))
    loss_df.set_index("Cell", inplace=True)

    c_list = sns.color_palette('hls', len(uncleared_alarms))
    plt.figure(figsize=[20, 8])
    total_loss = 0
    all_cell_list = loss_df.index.to_list()
    alarm_cell_related_dict = get_alarm_cell_related_dict(uncleared_alarms)
    alarm_region = get_all_alarm_region(uncleared_alarms)

    with tqdm(total=len(all_cell_list)) as pbar:
        for cell in all_cell_list:
            pm_file = cell.replace('-', '_') + ".csv"
            if pm_file not in file_list:
                pbar.update()
                continue
            if loss_df.loc[cell, "Type"] == 'alarm':
                df = pd.read_csv(os.path.join(alarm_cell_csv_path, cell + '.csv'))
            else:
                df = pd.read_csv(os.path.join(neighbor_cell_csv_path, cell + '.csv'))
            df["Start Time"] = pd.to_datetime(df["Start Time"])
            df.set_index("Start Time", inplace=True)
            kpi_datetime = df.index.to_pydatetime()

            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.plot(kpi_datetime, df["Actual-" + TRAFFIC_KPI_LIST[0]], label="actual")
            plt.plot(kpi_datetime, df["Predict-" + TRAFFIC_KPI_LIST[0]], '--', label="predict")
            loss = loss_df.loc[cell, TRAFFIC_KPI_LIST[0]]
            plt.title(
                "{} Cell:{}\nAlarm Name:{}".format(loss_df.loc[cell, "Type"].title(), cell, loss_df.loc[cell, "Name"]),
                fontsize=20)
            xt = pd.date_range(df.index[0], df.index[-1], freq='d')
            xt = [xt[i] for i in range(len(xt)) if i % 2 == 0]
            plt.xticks(xt)

            ax.axvspan(interest_time[0], e + timedelta(minutes=-1), alpha=0.3, facecolor='black', label='time of interest:{}-{}'.format(interest_time[0], e + timedelta(minutes=-1)))
            ax.axvline(interest_time[0], c='r', ls="--")
            ax.axvline(e + timedelta(minutes=-1), c='r', ls="--")

            ds = pd.date_range(interest_time[0], interest_time[1])
            mid_date_x = ds[len(ds) // 2]
            mid_loss_y = max(df["Predict-" + TRAFFIC_KPI_LIST[0]].max(), df["Actual-" + TRAFFIC_KPI_LIST[0]].max()) / 2
            plt.annotate(r'time of interest',
                         xy=(mid_date_x, mid_loss_y), xycoords='data',
                         xytext=(50, 50), textcoords='offset points', fontsize=16,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

            if cell in alarm_cell_list:
                # 如果告警关联到该告警小区，这里可能会出现和专家不一致现象，
                # 就是专家标注出的是告警小区，实际在通过计算告警关联的时候会没关联到
                if cell in alarm_cell_related_dict:
                    for i, a in enumerate(alarm_cell_related_dict[cell]):
                        leg = "Alarm: {}, Occurred On:{}, Cleared On (NT):{}".format(a['Name'].title(),
                                                                                     a['Occurred On (NT)'],
                                                                                     a['Cleared On (NT)'])
                        ax.axvspan(a['Occurred On (NT)'], a['Cleared On (NT)'], alpha=0.3, facecolor=c_list[i], label=leg)
                else:
                    leg = "Alarm: None"
                    ax.axvspan('2021-01-01 00:00:00', '2021-01-01 01:00:00', alpha=0, facecolor=c_list[0], label=leg)
            else:
                leg = "All Alarm Periods"
                ax.axvspan(alarm_region[0], alarm_region[1], alpha=0.3, label=leg)

            # yy = max(df["Actual-" + TRAFFIC_KPI_LIST[0]].max(), df["Predict-" + TRAFFIC_KPI_LIST[0]].max()) * 0.9
            # plt.text(df.index[-36], yy, "loss:{}".format(round(loss, 2)), fontsize=30, color='r', label=leg)
            plt.legend(loc=2)
            if cell in alarm_cell_list:
                plt.savefig(os.path.join(alarm_cell_fig_path, cell + '.png'))
            else:
                plt.savefig(os.path.join(neighbor_cell_fig_path, cell + '.png'))
            plt.clf()
            pbar.update()
            total_loss += loss


if __name__ == '__main__':
    # visualize(10)
    # case_list = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    case_list = [4, 13]
    for case_idx in case_list:
        visualize(case_idx)
