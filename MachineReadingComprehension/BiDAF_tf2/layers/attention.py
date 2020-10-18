# -*-coding:utf-8 -*-

'''
@File       : attention.py
@Author     : HW Shen
@Date       : 2020/10/18
@Desc       :
'''

import tensorflow as tf


class C2QAttention(tf.keras.layers.Layer):
    """ Context to Question 的 注意力实现 """

    def call(self, similarity, qencode):
        qencode = tf.expand_dims(qencode, axis=1)

        # 对输入的similarity结果在最有一个维度上进行softmax()处理
        c2q_att = tf.keras.activations.softmax(x=similarity, axis=-1)
        c2q_att = tf.expand_dims(input=c2q_att, axis=-1)  # 在尾部增加一个维度
        # 对倒数第二个维度部分进行“去维度求和”
        c2q_att = tf.math.reduce_sum(input_tensor=c2q_att * qencode, axis=-2)

        return c2q_att


class Q2CAttention(tf.keras.layers.Layer):
    """ Question to Context 的注意力实现 """

    def call(self, similarity, cencode):

        # 在axis=-1维度上求最大值
        max_similarity = tf.math.reduce_max(input_tensor=similarity, axis=-1)
        c2q_att = tf.keras.activations.softmax(x=max_similarity) # 对最大值进行softmax()处理
        c2q_att = tf.expand_dims(input=c2q_att, axis=-1)  # 在尾部增加一个维度

        # reduce_sum()为压缩求和，用于降维, 这里是对-2维度上的进行处理
        weighted_sum = tf.math.reduce_sum(input_tensor=c2q_att * cencode, axis=-2)
        weighted_sum = tf.expand_dims(input=weighted_sum, axis=1)  # 在尾部增加一个维度

        num_repeat = cencode.shape[1]

        # 用于在同一维度上进行复制，multiples表示复制次数
        q2c_att = tf.tile(input=weighted_sum, multiples=[1, num_repeat, 1])

        return q2c_att
