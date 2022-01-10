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


def weight_variable(shape):
    # generate noralizated distribution
    """
    产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
    shape，生成张量的维度
    mean，均值
    stddev，标准差
	"""
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)
