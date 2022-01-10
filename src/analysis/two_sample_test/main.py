import os
import pandas as pd

from two_sample_test import KNNTwoSampleTest
from info_extractor import InfoExtractor
from data_loader import DataLoader
from src.analysis.two_sample_test.utils import computeRightK


def twoSampleTest_oneCell(kpi_df, cfg_df, hyperparameters):
    """
    :param kpi_df: pandas dataframe(all kpis)
    :param cfg_df: pandas dataframe(all parameters)
    :hyperparameters: dict: {
        'alpha': the threshold of confident coefficient
        'r': the nearest_neighbor_num
    }
    """
    info_ext = InfoExtractor()
    kpi_names = info_ext.getKPINames(kpi_df)
    eventOccurDict_perParam = info_ext.getEventOccurTimes(cfg_df)

    result_dict_correlation = {}
    result_dict_confident = {}
    result_dict_correlation['kpi_name'] = kpi_names  # 第一列为kpi_name, 之后每一列是对应的参数与哪些kpi相关
    result_dict_confident['kpi_name'] = kpi_names

    # params->kpis
    for param_name in eventOccurDict_perParam.keys():
        param_correlated_result = []
        param_confident_result = []
        if len(eventOccurDict_perParam[param_name]) < 1:
            # NOTE: 说明该参数没有发生过变化，不用分析它与kpi的相关性
            break
        for kpi_name in kpi_names:
            kpi_rear_data = []
            kpi_timeseries = list(kpi_df.loc[:, kpi_name])
            # NOTE: 利用自相关函数计算该KPI应选取的k值
            right_k = computeRightK(kpi_df.loc[:, kpi_name])
            knn_tst = KNNTwoSampleTest(_alpha=hyperparameters['alpha'], _r=hyperparameters['r'], _k=right_k)
            # NOTE: 针对每个KPI，根据ACF计算所得的k来进行rear sample和random sample
            kpi_rear_data = info_ext.getRearDataPerParamPerKPI(kpi_df, kpi_name, eventOccurDict_perParam[param_name],
                                                               subseries_windowsz=right_k)
            flag, confident = knn_tst.excuteTwoSampleTest(eventOccurDict_perParam[param_name], kpi_rear_data,
                                                          kpi_timeseries)
            param_correlated_result.append(flag)
            param_confident_result.append(confident)
        result_dict_correlation[param_name] = param_correlated_result
        result_dict_confident[param_name] = param_confident_result

    result_df_correlation = pd.DataFrame(result_dict_correlation)
    result_df_confident = pd.DataFrame(result_dict_confident)
    return result_df_correlation, result_df_confident


def run():
    dl = DataLoader()
    # knn-two-sample-test hyperparameters
    hyperparameters = {
        'alpha': 2.58,
        'r': 5
    }

    # TODO: 这里以一个小区为例，批量化操作的时候可以遍历所有小区
    # NOTE: data_dir表示two_sample_test 存放数据的目录，具体使用时需要将KPI和配合数据放在该目录下
    data_dir = 'src/analysis/two_sample_test/data'
    kpi_file = os.path.join(data_dir, 'KPI', '0.csv')
    config_file = os.path.join(data_dir, '配置', '0.csv')

    kpi_df = dl.loadKPI(kpi_file)
    cfg_df = dl.loadCFG(config_file)

    result_df_correlation, result_df_confident = twoSampleTest_oneCell(kpi_df, cfg_df, hyperparameters)

    # output result
    result_dir = os.path.join('result_cell0')
    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)
    result_file_corr = os.path.join(result_dir, 'two-sample_test_correlation_alpha={}_r={}_k={}.csv'.format(
        hyperparameters['alpha'], hyperparameters['r'], 'auto-select'
    ))
    result_file_conf = os.path.join(result_dir, 'two-sample_test_confident_alpha={}_r={}_k={}.csv'.format(
        hyperparameters['alpha'], hyperparameters['r'], 'auto-select'
    ))

    result_df_correlation.to_csv(result_file_corr)
    result_df_confident.to_csv(result_file_conf)


if __name__ == '__main__':
    run()
