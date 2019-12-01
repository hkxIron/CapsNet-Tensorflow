"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import numpy as np
import tensorflow as tf

from config import cfg


epsilon = 1e-9


class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing: #没有路由
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                '''
                # version 1, computational expensive
                capsules = []
                for i in range(self.vec_len): # capsule向量本身的维度
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)): # variablea_scope名字不一样
                        # input:[batch, 20, 20, 256]
                        # caps_i:[batch, (20-9+0)//2+1=6, 6, 32],
                        # 可以看到向量8个维度之间,没有共享任何参数
                        caps_i = tf.contrib.layers.conv2d(input, 
                                                          self.num_outputs, # 32
                                                          self.kernel_size, # 9
                                                          self.stride, # 2
                                                          padding="VALID", 
                                                          activation_fn=None)
                        # caps_i:[batch, 6*6*32 = 1152, 1, 1]
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        # list of caps_i
                        capsules.append(caps_i)
                assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1] 
                # capsules: [batch, 1152, 8, 1]
                capsules = tf.concat(capsules, axis=2)
                '''

                # version 2, equivalent to version 1 but higher computational
                # efficiency.
                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice
                #---------------------
                # capsules:[batch, (20-9+0)//2+1=6, 6, 32*8],
                capsules = tf.contrib.layers.conv2d(input,
                                                    self.num_outputs * self.vec_len, # 32 * 8
                                                    self.kernel_size, # 9
                                                    self.stride, # 2
                                                    padding="VALID",
                                                    activation_fn=tf.nn.relu)
                # capsules = tf.contrib.layers.conv2d(input,
                #                                    self.num_outputs * self.vec_len,
                #                                    self.kernel_size,
                #                                    self.stride,padding="VALID",
                #                                    activation_fn=None)
                # capsules:[batch, caps_num=6*6*32=1152, vec_dim=8, 1],
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return capsules
            else:
                raise RuntimeError("invalid params!")

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer

                # input:[batch, caps_in_num=1152, vec_dim=8, 1]
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1)) # 或者tf.expand_dims()
                print("fc input:{}".format(self.input.shape))

                with tf.variable_scope('routing'):
                    # about the reason of using 'batch_size', see issue #21
                    # https://github.com/naturomics/CapsNet-Tensorflow/issues/21
                    """
                    , b_IJ will be re-init to 0 at each batch, it's not shared between batches. 
                    Someone(not me) had done experiments about this problem and he told me it does work in that way.
                    """
                    # b_IJ: [batch_size, caps_in_num, caps_out_num, 1, 1], 每个batch里bij会置0
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    # routing之后,得到高层语义向量
                    # input:[batch_size, caps_in_num=1152, 1, vec_dim=8, 1]
                    # capsules:[batch_size, caps_in_num=1, caps_out_num=10, caps_out_dim=16, 1]
                    capsules = routing(self.input, b_IJ)
                    # capsules:[batch_size, caps_out_num=10, caps_out_dim=16, 1]
                    capsules = tf.squeeze(capsules, axis=1)

            return capsules

def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, caps_in_num=1152, caps_out_num=10, len_u_i=caps_in_vec_dim=8, len_v_j=caps_out_vec_dim=16]
    W = tf.get_variable('Weight',
                        shape=(1, 1152, 10, 8, 16),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input:[batch_size, caps_in_num=1152, 1, vec_dim=8, 1]
    # input tile => [batch_size, 1152, 10, 8, 1]
    input = tf.tile(input, [1, 1, 10, 1, 1]) # 将input在axis=2上复制10份

    # W: [1, num_caps_i=1152, num_caps_j=10, len_u_i=8, len_v_j=16]
    # W tile => [batch_size, 1152, 10, 8, 16]
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch

    # 1.将输入向量转换到输出向量的空间维度(或者称为变换)
    #     W:[batch_size, caps_in_num=1152, caps_out_num=10, caps_in_dim=8, caps_out_dim=16]
    # input:[batch_size, caps_in_num=1152, caps_out_num=10, caps_in_dim=8, 1]
    # u_hat:[batch_size, caps_in_num=1152, caps_out_num=10, caps_out_dim=16, 1]
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat;
    # in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):

            # b_IJ: [batch_size, caps_in_num=1152, caps_out_num=10, 1, 1], 每个batch里bij会置0
            # line 4: => [batch_size, 1152, 10, 1, 1], c_ij = softmax(bij, axis=output_dim_j)
            # c_IJ: [batch_size, caps_in_num=1152, caps_out_num=10, 1, 1],归一化概率
            c_IJ = tf.nn.softmax(b_IJ, axis=2) # 输入caps对输出caps的预测归一化概率
            #print("c_IJ:", c_IJ.shape)
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:

                print("c_IJ:", c_IJ.shape, " u_hat:", u_hat.shape) # c_IJ: (5, 1152, 10, 1, 1)  u_hat: (5, 1152, 10, 16, 1)
                # c_IJ: [batch_size, caps_in_num=1152, caps_out_num=10, 1, 1],归一化概率
                # u_hat:[batch_size, caps_in_num=1152, caps_out_num=10, caps_out_dim=16, 1]
                # line 5: s_j = sum_i {c_ij*u_hat{j|i}}
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                # s_J:[batch_size, caps_in_num=1152, caps_out_num=10, caps_out_dim=16, 1]
                s_J = tf.multiply(c_IJ, u_hat) # multiply会自动复制未对齐的行
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                # s_J:[batch_size, caps_in_num=1, caps_out_num=10, caps_out_dim=16, 1], 对input_dim进行求和
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                # v_J:[batch_size, caps_in_num=1, caps_out_num=10, caps_out_dim=16, 1]
                v_J = squash(s_J) # 这个是高层语义的向量
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped) # 中间的迭代一直用的无梯度u
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7: bij=bij+u_hat{j|i}*vj, bij为标量,u_hat*vj为向量点乘
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]

                # v_J:[batch_size, caps_in_num=1, caps_out_num=10, caps_out_dim=16, 1]
                # v_J_tiled:[batch_size, caps_in_num=1152, caps_out_num=10, caps_out_dim=16, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                # u_hat_stopped:[batch_size, caps_in_num=1152, caps_out_num=10, caps_out_dim=16, 1]
                # u_produce_v:[batch_size, caps_in_num=1152, caps_out_num=10, 1, 1], 更新所有的高层向量对低层向量的权重系数
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v # 更新bIJ

    return v_J


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    f(x) = |x|^2/(1+|x|^2)*x/|x|
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), axis=-2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed
