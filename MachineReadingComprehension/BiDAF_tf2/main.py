# -*-coding:utf-8 -*-

'''
@File       : main.py
@Author     : HW Shen
@Date       : 2020/9/27
@Desc       :
'''

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
from MachineReadingComprehension.BiDAF_tf2 import layers
from MachineReadingComprehension.BiDAF_tf2.layers.char_cnn import CharCNN
from MachineReadingComprehension.BiDAF_tf2 import preprocess
import numpy as np

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class BiDAF(object):
    """
    双向注意流模型
        :param clen: context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 *2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
    """
    def __init__(self, clen, qlen, emb_size,
                 max_features=5000,
                 num_highway_layers=2,
                 encoder_dropout=0,
                 num_decoders=2,
                 decoder_dropout=0,
    ):

        self.clen = clen
        self.qlen = qlen
        self.max_features = max_features
        self.emb_size = emb_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    def build_model(self):
        """ 构建模型 """

        # 1 embedding 层
        # TODO：homework：使用 glove word embedding（或自己训练的 w2v） 和 char-CNN  embedding
        # Glove word embedding
        self.glove_word_embed = get_glove_vec(BASE_DIR+"/data/glove_vectors.txt", BASE_DIR + "/data/vocab.txt")

        # CNN char embeddig
        CharCNN()

        embedding_layer = tf.keras.layers.Embedding(input_dim=self.max_features,
                                                    output_dim=self.emb_size,
                                                    embeddings_initializer='uniform',
                                                    )

        c_inp = tf.keras.layers.Input(shape=(self.clen,), name='context_input')  # 上下文输入层
        q_inp = tf.keras.layers.Input(shape=(self.qlen,), name='question_input')  # 问题输入层

        c_emb = embedding_layer(c_inp)  # 上下文嵌入层
        q_emb = embedding_layer(q_inp)  # 问题嵌入层

        for i in range(self.num_highway_layers):

            # 使用两层高速神经网络
            highway_layer = layers.Highway(name=f'Highway{i}')
            c_highway = tf.keras.layers.TimeDistributed(layer=highway_layer, name=f'CHighway{i}')
            q_highway = tf.keras.layers.TimeDistributed(layer=highway_layer, name=f'QHighway{i}')
            c_emb = c_highway(c_emb)
            q_emb = q_highway(q_emb)

        # 2.上下文嵌入层 context_embedding
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        c_encode = encoder_layer(c_emb)  # 编码context
        q_encode = encoder_layer(q_emb)  # 编码question

        # 3.注意流层 attention flow
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([c_encode, q_encode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, q_encode)
        q2c_att = q2c_att_layer(similarity_matrix, c_encode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(c_encode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([c_encode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inp = [c_inp, q_inp]  # 拼接context和question的input

        self.model = tf.keras.models.Model(inp, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)  # 优化器选用 Adadelta
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )


from tqdm import tqdm


def get_glove_vec(glove_path, voc_path):
    """  """
    # glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    # sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    # total = sizes[glove_size]

    word2id = {}
    for line in open(voc_path, encoding='utf-8').readlines():
        word, word_id = line.split()
        word2id[word] = word_id

    glove_vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word2id:
                glove_vec_dict[word] = vector
            elif word.capitalize() in word2id:
                glove_vec_dict[word.capitalize()] = vector
            elif word.lower() in word2id:
                glove_vec_dict[word.lower()] = vector
            elif word.upper() in word2id:
                glove_vec_dict[word.upper()] = vector

    return glove_vec_dict


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)


if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    train_c, train_q, train_y = ds.get_dataset('./data/squad/train-v1.1.json')
    test_c, test_q, test_y = ds.get_dataset('./data/squad/dev-v1.1.json')

    print(train_c.shape, train_q.shape, train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=50,
        max_features=len(ds.charset)
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_c, train_q], train_y,
        batch_size=64,
        epochs=10,
        validation_data=([test_c, test_q], test_y)
    )

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)