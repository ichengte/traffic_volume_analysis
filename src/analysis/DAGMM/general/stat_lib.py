import math
import tensorflow.compat.v1 as tf


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


class GaussianMixtureModeling:
    def __init__(self, gmm_config):
        self.num_mixture = gmm_config[0]
        self.num_dim = gmm_config[1]
        self.num_dynamic_dim = gmm_config[2]

    def eval(self, x, p):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Cluster distribution phi: [num_mixture]
        phi = tf.reduce_mean(p, 0)
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])
        # mixture mean: [num_mixture, num_dim]
        p_t = tf.reshape(p, shape=[-1, self.num_mixture, 1])
        z_p = tf.reduce_sum(p_t, 0)
        mixture_mean = tf.reduce_sum(x_t * p_t, 0) / z_p
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        mixture_cov = tf.reduce_sum(z_t * p_t, 0) / z_p
        mixture_dev = mixture_cov ** 0.5
        # probability density evaluation
        z_norm = tf.reduce_sum(z_t / mixture_cov, 2)
        mixture_dev_det = tf.reduce_prod(mixture_dev, 1)
        t1 = tf.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # Likelihood
        tmp = phi * (t1 / t2)
        likelihood = tf.reduce_sum(tmp, 1)
        energy = tf.reduce_mean(-tf.log(likelihood + 1e-12))
        pen_dev = tf.reduce_sum(1.0 / mixture_cov[:, 0:self.num_dynamic_dim])
        return energy, pen_dev, likelihood, phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        """
        Arguments specification
        :param x: input dd with m dimensions (compressed code in AE); [batch_size, num_dim]
        :param p: mixture assignment for x; [batch_size, num_mixture]
        This method uses diagonal covariance matrix
        :return:
        """
        # Augmenting input: [batch_size, num_mixture, num_dim]
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])
        # mixture diagonal covariance: [num_mixture, num_dim]
        z_t = (x_t - mixture_mean) ** 2
        # probability density evaluation
        z_norm = tf.reduce_sum(z_t / mixture_cov, 2)
        mixture_dev_det = tf.reduce_prod(mixture_dev, 1)
        t1 = tf.exp(-0.5 * z_norm)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * mixture_dev_det
        # Likelihood
        tmp = phi * t1 / t2
        likelihood = tf.reduce_sum(tmp, 1)
        return likelihood

    @staticmethod
    def help():
        print("The copyright of source code belongs to NECLA. Only for non-commercial evaluation.")
