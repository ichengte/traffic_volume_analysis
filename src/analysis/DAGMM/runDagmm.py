import tensorflow.compat.v1 as tf
import architecture.Dagmm as dagmm
import sys, os
import general.stat_lib as lib

tf.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
Copyright (C) 2016 NEC Laboratories America, Inc. ("NECLA")

This software and any and all related files/code/information is provided by NECLA to for 
non-commercial evaluation or research purposes subject to terms in a License agreement the Recipient has agreed to 
by Recipient's signature.

The license restriction includes, among other limitations, the Recipient to only evaluate this software and 
redistribute information related to this software only in the form of technical publications/papers, with no rights to 
assign a license to third parties or redistribute the software to others. 

IN NO EVENT SHALL NEC BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES 
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF NEC HAS BEEN ADVISED OF THE POSSIBILITY OF 
SUCH DAMAGE.

NEC SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND NEC HAS NO OBLIGATION 
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

THE LICENSE FROM NEC FOR THE SOFTWARE REQUIRES THAT LICENSEE COMPLY WITH ANY AND ALL UNDERLYING COPYRIGHTS AND LICENSE 
RIGHTS IN THE SOFTWARE BY THIRD PARTIES. 
"""


def thyroid_evaluate():
    filename = "demo/thyroid.pp.csv.gz"
    cnet_config = [6,
                   12,
                   4,
                   1
                   ]
    enet_config = {'general': [3, 2, 1],
                   'layer_1': [10]}
    precision = 0
    recall = 0
    f1 = 0
    rounds = 20
    epoch = 20000
    threshold = 0.025
    batch_size = 1024
    step = 100
    for k in range(rounds):
        tf.reset_default_graph()
        machine = dagmm.DAGMM(6, cnet_config, enet_config)
        data = machine.get_thyroid_data(filename)
        p, r, f = machine.run(data, epoch, threshold, batch_size, step)
        precision = precision + p
        recall = recall + r
        f1 = f1 + f
    print("avg. precision: %g; avg. recall: %g; avg. f1: %g" % (precision / rounds, recall / rounds, f1 / rounds))


def arrhythmia_evaluate():
    filename = "demo/heart.pp.csv.gz"
    cnet_config = [274,
                   10,
                   2
                   ]
    enet_config = {'general': [4, 2, 2],
                   'layer_1': [10]}
    precision = 0
    recall = 0
    f1 = 0
    rounds = 20
    epoch = 10000
    threshold = 0.15
    batch_size = 128
    step = 100
    for k in range(rounds):
        tf.reset_default_graph()
        machine = dagmm.DAGMM(274, cnet_config, enet_config)
        data = machine.get_heart_data(filename)
        p, r, f = machine.run(data, epoch, threshold, batch_size, step)
        precision = precision + p
        recall = recall + r
        f1 = f1 + f
    print("avg. precision: %g; avg. recall: %g; avg. f1: %g" % (precision / rounds, recall / rounds, f1 / rounds))


def kddcuprev_evaluate():
    filename = "src/analysis/DAGMM/demo/kddcup99-10.data.pp.csv.gz"
    cnet_config = [120,
                   60,
                   30,
                   10,
                   1]
    enet_config = {'general': [3, 2, 1],
                   'layer_1': [10]}
    precision = 0
    recall = 0
    f1 = 0
    rounds = 20
    epoch = 20
    # rounds = 1
    # epoch = 50
    threshold = 0.2
    batch_size = 1024
    step = 5
    for k in range(rounds):
        tf.reset_default_graph()  # 无论执行多少次生成的张量始终不变, 换句话说就是：tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形。
        machine = dagmm.DAGMM(120, cnet_config, enet_config)
        data = machine.get_kddcuprev_data(filename)
        p, r, f = machine.run(data, epoch, threshold, batch_size, step)
        precision = precision + p
        recall = recall + r
        f1 = f1 + f
    print("avg. precision: %g; avg. recall: %g; avg. f1: %g" % (precision / rounds, recall / rounds, f1 / rounds))


def kddcup_evaluate():
    filename = "demo/kddcup99-10.data.pp.csv.gz"
    cnet_config = [120,
                   60,
                   30,
                   10,
                   1]
    enet_config = {'general': [3, 4, 1],
                   'layer_1': [10]}
    precision = 0
    recall = 0
    f1 = 0
    rounds = 1
    # epoch = 200
    epoch = 20
    threshold = 0.2
    batch_size = 1024
    step = 1
    for k in range(rounds):
        tf.reset_default_graph()
        machine = dagmm.DAGMM(120, cnet_config, enet_config)
        data = machine.get_kddcup_data(filename)
        p, r, f = machine.run(data, epoch, threshold, batch_size, step)
        precision = precision + p
        recall = recall + r
        f1 = f1 + f
    print("avg. precision: %g; avg. recall: %g; avg. f1: %g" % (precision / rounds, recall / rounds, f1 / rounds))


def gts_evaluate():
    class Data:
        def __init__(self, train_data, test_data, test_label, cell, start_time, alarm_cell_list, neighbor_cell_list,
                     uncleared_alarms, case_idx):
            self.train_data = train_data
            self.test_data = test_data
            self.test_label = test_label
            self.cell = cell
            self.start_time = start_time
            self.alarm_cell_list = alarm_cell_list
            self.neighbor_cell_list = neighbor_cell_list
            self.case_idx = case_idx
            self.uncleared_alarms = uncleared_alarms
            self.num_train_points = len(train_data)
            self.num_test_points = len(test_data)

    cnet_config = [62,
                   32,
                   16,
                   8,
                   1]
    enet_config = {'general': [3, 2, 1],
                   'layer_1': [10]}
    machine = dagmm.DAGMM(62, cnet_config, enet_config)
    gts_data = machine.get_gts_data(14)
    train_data_dict, test_data_dict, test_label_dict, start_time, alarm_cell_list, neighbor_cell_list, uncleared_alarms, case_idx = gts_data.train_data, gts_data.test_data, gts_data.test_label, gts_data.start_time, gts_data.alarm_cell_list, gts_data.neighbor_cell_list, gts_data.uncleared_alarms, gts_data.case_idx
    for cell in train_data_dict:
        precision = 0
        recall = 0
        f1 = 0
        rounds = 1
        epoch = 10000
        batch_size = 500
        step = 5000
        train_data, test_data, test_label = train_data_dict[cell], test_data_dict[cell], test_label_dict[cell]
        threshold = test_label.sum() / len(test_label)
        print('threshold:{}'.format(threshold))

        # tf.reset_default_graph()
        data = Data(train_data, test_data, test_label, cell, start_time, alarm_cell_list, neighbor_cell_list,
                    uncleared_alarms, case_idx)
        p, r, f = machine.run(data, epoch, threshold, batch_size, step)
        precision = precision + p
        recall = recall + r
        f1 = f1 + f
        if cell in gts_data.alarm_cell_list:
            print("Alarm Cell:", cell, "avg. precision: %g; avg. recall: %g; avg. f1: %g" % (
            precision / rounds, recall / rounds, f1 / rounds))
        else:
            print("Neighbor Cell:", cell, "avg. precision: %g; avg. recall: %g; avg. f1: %g" % (
            precision / rounds, recall / rounds, f1 / rounds))


if __name__ == '__main__':
    gts_evaluate()
    # kddcuprev_evaluate()

    # if len(sys.argv) == 3 and sys.argv[1] == "--dataset":
    #     if sys.argv[2] == "kddcup":
    #         kddcup_evaluate()
    #     elif sys.argv[2] == "thyroid":
    #         thyroid_evaluate()
    #     elif sys.argv[2] == "arrhythmia":
    #         arrhythmia_evaluate()
    #     elif sys.argv[2] == "kddcuprev":
    #         kddcuprev_evaluate()
    #     elif sys.argv[2] == "copyright":
    #         lib.GaussianMixtureModeling.help()
    #     else:
    #         print("Unknown dataset. terminated...")
    #     latent_var_list = tf.get_collection('latent')
    #     with open('new.txt', 'w') as f:
    #         for i in latent_var_list:
    #             f.write(str(i) + '\n')
    # else:
    #     print("Invalid command. terminated...")
