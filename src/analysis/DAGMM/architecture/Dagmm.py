import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import data.kddcup10 as kddcup
import data.thyroid as thyroid
import data.heart as heart
import data.kddcup10rev as kddcuprev
import general.estimation_net as enet
import general.compression_net as cnet
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import seaborn as sns
import src.utils as tools

from src.analysis.DAGMM.data.gts import GTSData

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


def get_key(item):
    return item[0]


class DAGMM:
    def __init__(self, num_input_dim, cnet_config, enet_config):
        self.num_input_dim = num_input_dim
        self.c_net = cnet.CompressionNet(cnet_config)
        self.e_net = enet.EstimationNet(enet_config)

    @staticmethod
    def get_gts_data(case_idx):
        gts_data = GTSData(case_idx)
        gts_data.get_clean_training_testing_data()
        return gts_data

    @staticmethod
    def get_kddcup_data(input_file):
        kddcup_data = kddcup.Kddcup(input_file)
        kddcup_data.get_clean_training_testing_data(0.5)
        return kddcup_data

    @staticmethod
    def get_thyroid_data(input_file):
        thyroid_data = thyroid.ThyroidData(input_file)
        thyroid_data.get_clean_training_testing_data(0.5)
        return thyroid_data

    @staticmethod
    def get_heart_data(input_file):
        heart_data = heart.HeartData(input_file)
        heart_data.get_clean_training_testing_data(0.5)
        return heart_data

    @staticmethod
    def get_kddcuprev_data(input_file):
        kddcuprev_data = kddcuprev.KddcupDataRev(input_file)
        kddcuprev_data.get_clean_training_testing_data(0.5)
        return kddcuprev_data

    @staticmethod
    def minmax_normalization(x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        return norm_x

    @staticmethod
    def correctness(predict, truth, num_test_points):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(num_test_points):
            if predict[i, 0] > 0.5:
                if truth[i, 0] > 0.5:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if truth[i, 0] > 0.5:
                    fn = fn + 1
                else:
                    tn = tn + 1
        return tp, fp, tn, fn

    def accuracy(self, predict_lh, num_test_points, test_y, outlier_ratio):
        tmp = []
        for i in range(num_test_points):
            p = (predict_lh[i], i)
            tmp.append(p)
        tmp.sort(key=get_key)
        predict = np.zeros([num_test_points, 1])
        # num_tag = int(num_test_points * outlier_ratio)
        # for i in range(num_tag):
        #     p = tmp[i]
        #     idx = p[1]
        #     predict[idx, 0] = 1
        for i in range(len(tmp)):
            p = tmp[i]
            idx = p[1]
            if p[0] < 0.1:
                predict[idx, 0] = 1
        tp, fp, tn, fn = self.correctness(predict, test_y, num_test_points)
        if tp + fp == 0:
            precision = np.nan
        else:
            precision = float(tp) / (tp + fp)

        if tp + fn == 0:
            recall = np.nan
        else:
            recall = float(tp) / (tp + fn)

        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = np.nan
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1, predict

    def run(self, data, num_epoch, threshold, bsize, step_size):
        sess = tf.InteractiveSession()
        # Data
        train_x = data.train_data
        test_x = data.test_data
        test_y = data.test_label
        base = np.concatenate([train_x, test_x], 0)

        num_train_points = data.num_train_points
        num_test_points = data.num_test_points

        train_norm_x = self.minmax_normalization(train_x, base)
        test_norm_x = self.minmax_normalization(test_x, base)
        # train_norm_x = train_x
        # test_norm_x = test_x

        # Setup
        train_x_v = tf.placeholder(dtype=tf.float64, shape=[None, self.num_input_dim])
        test_x_v = tf.placeholder(dtype=tf.float64, shape=[None, self.num_input_dim])
        keep_prob = tf.placeholder(tf.float64)
        x_b = tf.train.shuffle_batch(
            [train_norm_x],
            batch_size=bsize,
            num_threads=4,
            capacity=50000,
            enqueue_many=True,
            min_after_dequeue=10000)
        # Autoencoder
        z_b, error_b = self.c_net.run(x_b)  # encoder output, reconstruct loss
        train_z, error = self.c_net.run(train_x_v)
        # testing
        # test_zo is the output of decoder
        # test_zc is the output of encoder
        test_z, test_zc, test_zo = self.c_net.test(test_x_v)  # e_net input, encoder output, c_net output
        tf.add_to_collection('latent', test_zc)
        # GMM Membership estimation
        loss_b, pen_dev_b, likelihood_b = self.e_net.run(z_b, keep_prob)
        # loss_b, e_net_reg_b = self.e_net.run(z_b, keep_prob)
        loss, pen_dev, likelihood = self.e_net.run(train_z, keep_prob)
        # loss, e_net_reg = self.e_net.run(train_z, keep_prob)
        model_phi, model_mean, model_dev, model_cov = self.e_net.model(train_z, keep_prob)
        # testing
        test_likelihood = self.e_net.test(test_z, model_phi, model_mean, model_dev, model_cov)
        # Train step
        obj = error_b + loss_b * 0.1 + pen_dev_b * 0.005
        obj_oa = error + loss * 0.1 + pen_dev * 0.005
        train_step = tf.train.AdamOptimizer(1e-4).minimize(obj)

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        epoch_tot = num_epoch
        num_step = num_train_points // bsize + 1
        for k in range(epoch_tot):
            for i in range(num_step):
                train_step.run(feed_dict={keep_prob: 0.5})
            if (k + 1) % step_size == 0:
                train_obj = obj_oa.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                train_err = error.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                train_loss = loss.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                train_dev = pen_dev.eval(feed_dict={keep_prob: 1.0, train_x_v: train_norm_x})
                print("Epoch %d: objective %g; error %g; mix loss %g; dev penalty %g"
                      % (k + 1, train_obj, train_err, train_loss, train_dev))
        predict_lh = test_likelihood.eval(feed_dict={train_x_v: train_norm_x, test_x_v: test_norm_x, keep_prob: 1.0})

        precision, recall, f1, predict = self.accuracy(predict_lh, num_test_points, test_y, threshold)

        coord.request_stop()
        coord.join(threads)
        sess.close()
        # print("Precision: %g; Recall %g; F1 %g" % (precision, recall, f1))

        # initial variable
        case_idx = data.case_idx
        cell = data.cell
        start_time = data.start_time
        alarm_cell_list = data.alarm_cell_list
        neighbor_cell_list = data.neighbor_cell_list
        kpi_data_list = np.array(data.test_data)[:, 41]
        kpi_time_list = pd.date_range(start_time, freq='H', periods=len(kpi_data_list))
        uncleared_alarms = data.uncleared_alarms

        # create figure
        plt.figure(figsize=[20, 8])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.plot(kpi_time_list, kpi_data_list, label="LTE_DL Traffic Volume(GB)")

        # scatter anomaly
        predict_idx = []
        for i, p in enumerate(predict):
            if p == 1:
                predict_idx.append(i)
        mask = np.array([np.nan] * len(kpi_data_list))
        mask[predict_idx] = 1
        plt.scatter(kpi_time_list, kpi_data_list * mask, label='Detected anomalies', c='r', s=10)

        # plot uncleared_alarm
        c_list = sns.color_palette('hls', len(uncleared_alarms))
        uncleared_alarms = uncleared_alarms.reset_index()
        for i, a in uncleared_alarms.iterrows():
            leg = 'Alarm Name: {}, Occurred On {}, Cleared On {}'.format(a['Name'], a['Occurred On (NT)'],
                                                                         a['Cleared On (NT)'])
            ax.axvspan(a['Occurred On (NT)'], a['Cleared On (NT)'], alpha=0.5, label=leg, facecolor=c_list[i])

        plt.legend(loc=2)
        # plot title
        if cell in alarm_cell_list:
            plt.title('Alarm Cell:{}'.format(cell))
        else:
            plt.title('Neighbor Cell:{}'.format(cell))
        tools.check_path('result_ppt/dagmm/case{}'.format(case_idx))
        plt.savefig('result_ppt/dagmm/case{}/{}.png'.format(case_idx, cell))
        plt.close()

        return precision, recall, f1
