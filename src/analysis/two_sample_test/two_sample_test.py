# -*- coding: UTF-8 -*-

from __future__ import division
import random
import math
from operator import itemgetter


class TwoSampleTest():
    """ 双样本检测基类
    
    Attributes:
    alpha: 置信度阈值(1.96->2.5%, 2.58->0.1%)
    """

    def __init__(self, _alpha=1.96) -> None:
        self.alpha = _alpha

    def _isCorrelated(self, significance):
        """
        :param significance: 置信度参数，当其值大于置信度阈值时，H1为真(存在关联关系)
        :return True / False
        """
        return significance > self.alpha


class KNNTwoSampleTest(TwoSampleTest):
    """ 基于KNN(最近邻算法)的双样本检测

    Attributes:
    r: KNN中的k, 表示最近邻的个数，经验规则下取样本总数p的自然对数
    k: 采样时间子序列的长度(每个样本为一个子时间序列)，可调超参数
    p: 样本总数(混合样本总数, |F|+|G|)
    confident_coefficient: 置信度，用于判断是否可以推翻原假设H0(F=G)
    """

    def __init__(self, _alpha=1.96, _r=5, _k=6) -> None:
        super().__init__(_alpha=_alpha)

        # NOTE: 这里的r和k是按经验值取的，实际情况可能需要调整:
        """
        r = 4~7表示样本总数在100~1000之间
        k = 6表示时间序列的自相关性延迟设置为6，具体可根据不同的时间序列进行调整
        """
        self.r = _r
        self.k = _k
        self.distance_measure = DistanceMeasure()
        self.confident_coefficient = 0
        # NOTE: 这里的p是随机取的初始值，后面做双样本检测时会修改
        self.p = 0

    def _mixData(self, event_occur_time, rear_data, timeseries, sample_multiple=3):
        """ 将两个分布(F、G)的样本混合在一起->mixset

        :param event_ocur_time: 事件发生的时间点列表(list，其中每一个元素为一个index)
        :param rear_data: 事件发生后紧接着的子时间序列样本集合(list，其中每个元素为一个子时间序列)
        :param timeseries: 原时间序列，用于随机抽样子时间序列
        :param sample_multiple: 随机采样倍数(随机采样样本个数与事件发生个数的比例)，可调整的超参数, 即randomSample_num // event_num

        :return: mixset, event_num, randomSample_num
        """
        if len(rear_data) < 1 or len(rear_data[0]) != self.k:
            if len(rear_data) == 0:
                raise ValueError("rear_data(F) is empty")
            else:
                raise ValueError(
                    "the window_size of rear data(F) and random sample data(G) must be the same, while k_rear={}, k_random={}".format(
                        len(rear_data[0]), self.k
                    ))

        mixset = []
        _event_num = len(event_occur_time)
        # HACK: 这里随机采样的样本个数的做法比较粗糙，进一步可尝试改进(有两个可以参考的实验比例: 1:1, 1:3)
        _randomSample_num = sample_multiple * _event_num

        r_event_num = 0
        r_randomSample_num = 0

        for i in range(len(event_occur_time)):
            data = rear_data[i]
            data.append('event')
            if len(data) > 1 and data not in mixset:
                mixset.append(data)
                r_event_num += 1

        while _randomSample_num > 0:
            end = random.randint(self.k, len(timeseries))
            start = end - self.k
            data = timeseries[start: end]
            data.append("random")
            _randomSample_num -= 1
            if len(data) > 1 and data not in mixset:
                mixset.append(data)
                r_randomSample_num += 1

        return mixset, r_event_num, r_randomSample_num

    def _computeConfidence(self, mixset, event_num, randomSample_num):
        """
        :return: confident coefficent
        """
        if event_num == 0 or randomSample_num == 0:
            raise ValueError("event_num and randomSample_num mustn't be zero")

        self.p = event_num + randomSample_num
        # NOTE: r = ln(p)向上取整
        self.r = int(math.log(self.p)) + 1
        # TODO: ensure the value of k
        mean = (event_num / self.p) ** 2 + (randomSample_num / self.p) ** 2
        stdDev = (event_num / self.p) * (randomSample_num / self.p) * (
                    1 + 4 * (randomSample_num / self.p) * (event_num / self.p))

        Trp = 0
        for j in range(len(mixset)):
            tempdic = {}
            for k in range(len(mixset)):
                if j == k:
                    continue
                # TODO: use DTW measure
                dis = self.distance_measure.euclideanDistance(mixset[j], mixset[k])
                tempdic.setdefault(k, dis)

            temp_list = sorted(tempdic.items(), key=itemgetter(1), reverse=False)[0:self.r]
            for k in temp_list:
                if mixset[j][-1] == mixset[k[0]][-1]:
                    Trp += 1

        Trp = float(Trp / (self.r * self.p))
        return abs((Trp - mean) / stdDev) * math.sqrt(self.r * self.p)

    def excuteTwoSampleTest(self, event_occur_time, rear_data, timeseries):
        """ 针对单种类型事件(event)和单条时间序列(timeseries)进行关联分析
        
        :param event_ocur_time: 事件发生的时间点列表(list，其中每一个元素为一个index)
        :param rear_data: 事件发生后紧接着的子时间序列样本集合(list，其中每个元素为一个子时间序列list)
        :param timeseries: 原时间序列，用于随机抽样子时间序列

        :return True(correlated) / False(not correlated), confident_coefficient
        """
        mixset, event_num, randomSample_num = self._mixData(
            event_occur_time, rear_data, timeseries
        )
        self.confident_coefficient = self._computeConfidence(
            mixset, event_num, randomSample_num
        )
        return self._isCorrelated(self.confident_coefficient), self.confident_coefficient


class DistanceMeasure():
    """ 计算样本间距离的类，这里只实现了欧氏距离(可补充：DTW, 基于密度等方法)

    """

    def __init__(self) -> None:
        pass

    def euclideanDistance(self, data1, data2):
        dis = 0
        # NOTE: 这里循环末尾不去最后一列是因为最后一列不是原始数值特征('event'和'random')
        for i in range(0, len(data1) - 1):
            dis += (data1[i] - data2[i]) ** 2
        dis = math.sqrt(dis)
        return dis
