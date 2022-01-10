import tensorflow.compat.v1 as tf

import general.param_init as pini


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


class CompressionNet:
    def __init__(self, config):
        self.num_dim = config
        self.code_layer = len(config)
        # Parameters in layers
        self.wi = []
        self.bi = []
        self.var_list1 = []
        self.var_list2 = []
        # Encode
        for i in range(0, len(self.num_dim)-1):
            # layer shape here ( num_dim[i] , num_dim[i+1] )
            w = pini.weight_variable([self.num_dim[i], self.num_dim[i+1]]) 
            b = pini.bias_variable([self.num_dim[i+1]])
            self.wi.append(w)
            self.bi.append(b)
            self.var_list1.append(w)
            self.var_list1.append(b)
        # Decode
        for i in range(1, len(self.num_dim)):
            j = len(self.num_dim)-i
            w = pini.weight_variable([self.num_dim[j], self.num_dim[j-1]])
            b = pini.bias_variable([self.num_dim[j-1]])
            self.wi.append(w)
            self.bi.append(b)
            self.var_list2.append(w)
            self.var_list2.append(b)

    def run(self, x):
        # Encode
        zi = x
        for i in range(len(self.wi)//2):
            if i < len(self.wi) // 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zc = zi
        # Decode
        for i in range(len(self.wi)//2, len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        # Relative distance
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # Assemble feature
        xo = tf.concat([zc, relative_dist, cos_sim], 1)
        # xo = tf.concat([zc, relative_dist], 1)
        error = tf.reduce_mean(dist)
        return xo, error

    def test(self, x):
        # Encode
        zi = x
        for i in range(len(self.wi) // 2):
            if i < len(self.wi) // 2 - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zc = zi
        #tf.add_to_collection('latent', zc)
        # Decode
        for i in range(len(self.wi) // 2, len(self.wi)):
            if i < len(self.wi) - 1:
                zj = tf.nn.tanh(tf.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = tf.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = tf.nn.l2_normalize(x, 1)
        normalize_zo = tf.nn.l2_normalize(zo, 1)
        cos_sim = tf.reduce_sum(tf.multiply(normalize_x, normalize_zo), 1, keep_dims=True)
        # Relative distance
        dist = tf.norm(x - zo, ord=2, axis=1, keep_dims=True)
        relative_dist = dist / tf.norm(x, ord=2, axis=1, keep_dims=True)
        # Assemble feature
        xo = tf.concat([zc, relative_dist, cos_sim], 1)
        # xo = tf.concat([zc, relative_dist], 1)
        return xo, zc, zo


