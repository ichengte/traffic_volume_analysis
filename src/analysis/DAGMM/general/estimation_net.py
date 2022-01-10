import general.param_init as pini
import tensorflow.compat.v1 as tf

import general.stat_lib as slib

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


class EstimationNet:
    def __init__(self, config):
        # DMM config
        self.dmm_config = config['general']
        self.input_dim = self.dmm_config[0]  # 3 : c_net_output, euclidean, cosine
        self.num_mixture = self.dmm_config[1]  # 2: num_gmm=2
        self.num_dynamic_dim = self.dmm_config[2]  #
        # Layer 1
        layer_1_config = config['layer_1']
        self.output_d_1 = layer_1_config[0]
        # 3->10
        self.w1 = pini.weight_variable([self.input_dim, self.output_d_1])
        self.b1 = pini.bias_variable([self.output_d_1])
        # Layer 2
        # layer_2_config = config['layer_2']
        # self.output_d_2 = layer_2_config[0]
        # 10->2
        self.w2 = pini.weight_variable([self.output_d_1, self.num_mixture])
        self.b2 = pini.bias_variable([self.num_mixture])
        # Mixture modeling
        self.gmm_config = [self.num_mixture, self.input_dim, self.num_dynamic_dim]
        self.gmm = slib.GaussianMixtureModeling(self.gmm_config)

    def run(self, x, keep_prob):
        # Mixture estimation network
        # Layer 1
        z1 = tf.nn.tanh(tf.matmul(x, self.w1) + self.b1)
        # Layer 2
        z1_drop = tf.nn.dropout(z1, keep_prob)
        p = tf.nn.softmax(tf.matmul(z1_drop, self.w2) + self.b2)
        # Log likelihood
        gmm_energy, pen_dev, likelihood, phi, _, mixture_dev, _ = self.gmm.eval(x, p)
        # k_dist = self.kmm.eval(x, p)
        # mixture_dev_0 = mixture_dev[:, 0]
        # _, prior_energy = self.inverse_gamma.eval(mixture_dev, phi)
        # train
        # energy = gmm_energy
        loss = gmm_energy
        # loss = k_dist
        # reg = 0
        # for w in self.var_list:
        #     reg = reg + tf.nn.l2_loss(w)
        # reg = tf.reduce_sum(phi * tf.log(phi+1e-12)) + 0.1*tf.reduce_mean(tf.reduce_sum(- p * tf.log(p + 1e-12), 1))
        return loss, pen_dev, likelihood
        # return loss, reg

    def model(self, x, keep_prob):
        # Mixture estimation network
        # Layer 1
        z1 = tf.nn.tanh(tf.matmul(x, self.w1) + self.b1)
        # Layer 2
        z1_drop = tf.nn.dropout(z1, keep_prob)
        p = tf.nn.softmax(tf.matmul(z1_drop, self.w2) + self.b2)
        # Log likelihood
        _, _, _, phi, mixture_mean, mixture_dev, mixture_cov = self.gmm.eval(x, p)
        return phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        likelihood = self.gmm.test(x, phi, mixture_mean, mixture_dev, mixture_cov)
        return likelihood
