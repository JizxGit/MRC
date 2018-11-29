import tensorflow as tf
import numpy as np


def matrix_multiplication(mat, weight):
    # mat [batch_size,seq_len,hidden_size] weight [size,size]
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    assert mat_shape[-1] == weight_shape[0]
    mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])
    mul = tf.matmul(mat_reshape, weight)
    return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])


a = tf.constant(np.random.randn(1,2,3))
b = tf.constant(np.random.randn(3))
c = tf.reduce_sum(a*b,axis=2)
b1=tf.expand_dims(b,axis=0)
b1=tf.expand_dims(b1,axis=2)
c1=tf.matmul(a,b1)

with tf.Session() as sess:
    print(a.eval())
    print("------")
    print(b.eval())
    print("------")
    print(c.eval())
    print(c1.eval())
    # print(cc.eval())
