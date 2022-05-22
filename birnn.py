# encoding: utf-8
#作者：韦访
#csdn：https://blog.csdn.net/rookie_wei
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from collections import Counter
from utils import dense_to_text, sparse_tuple_to_text

tf.compat.v1.disable_eager_execution()
"""
used to create a variable in CPU memory.
"""
def variable_on_cpu(name, shape, initializer):
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.compat.v1.get_variable(name=name, shape=shape, initializer=initializer)
    return var

class BiRNN():
    def __init__(self, features, contexts, batch_size, hidden, cell_dim, stddev, keep_dropout_rate, relu_clip, character, save_path, learning_rate):
        self.features = features
        self.batch_size = batch_size
        self.contexts = contexts
        self.hidden = hidden
        self.stddev = stddev
        self.keep_dropout_rate = keep_dropout_rate
        self.relu_clip = relu_clip
        self.cell_dim = cell_dim
        self.learning_rate = learning_rate

        # input 为输入音频数据，由前面分析可知，它的结构是[batch_size, amax_stepsize, features + (2 * features * contexts)]
        #其中，batch_size是batch的长度，amax_stepsize是时序长度，n_input + (2 * features * contexts)是MFCC特征数，
        #batch_size是可变的，所以设为None，由于每一批次的时序长度不固定，所有，amax_stepsize也设为None
        self.input = tf.compat.v1.placeholder(tf.float32, [None, None, features + (2 * features * contexts)], name='input')
       
        # label 保存的是音频数据对应的文本的系数张量，所以用sparse_placeholder创建一个稀疏张量
        self.label = tf.compat.v1.sparse_placeholder(tf.int32, name='label')

        #seq_length保存的是当前batch数据的时序长度
        self.seq_length = tf.compat.v1.placeholder(tf.int32, [None], name='seq_length')

        #keep_dropout则是dropout的参数
        self.keep_dropout = tf.compat.v1.placeholder(tf.float32, name='keep_dropout')
        
        self.network_init(self.input, character)
        self.loss_init()
        self.optimizer_init()
        self.accuracy_init()

        #创建会话
        self.sess = tf.compat.v1.Session()

        #需要保存模型，所以获取saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

        #模型保存地址
        self.save_path = save_path
        #如果该目录不存在，新建
        if os.path.exists(self.save_path) == False:
            os.mkdir(self.save_path)

        #初始化
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # 没有模型的话，就重新初始化
        cpkt = tf.compat.v1.train.latest_checkpoint(self.save_path)
        
        self.start_epoch = 0
        if cpkt != None:
            self.saver.restore(self.sess, cpkt)
            ind = cpkt.find("-")
            self.start_epoch = int(cpkt[ind + 1:])

    def get_property(self):
        return self.start_epoch

    def network_init(self, input, character):
        # batch_x_shape: [batch_size, amax_stepsize, n_input + 2 * n_input * contexts]
        batch_x_shape = tf.shape(input)
    
        # 将输入转成时间序列优先
        input = tf.transpose(input, [1, 0, 2])
        # 再转成2维传入第一层
        # [amax_stepsize * batch_size, n_input + 2 * n_input * contexts]
        input = tf.reshape(input, [-1, self.features + 2 * self.features * self.contexts])
        
        # 使用clipped RELU activation and dropout.
        # 1st layer
        with tf.name_scope('fc1'):
            b1 = variable_on_cpu('b1', [self.hidden], tf.random_normal_initializer(stddev=self.stddev))        
            h1 = variable_on_cpu('h1', [self.features + 2 * self.features * self.contexts, self.hidden],
                                tf.random_normal_initializer(stddev=self.stddev))
            layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(input, h1), b1)), self.relu_clip)
            layer_1 = tf.nn.dropout(layer_1, self.keep_dropout)
        
        # 2nd layer
        with tf.name_scope('fc2'):
            b2 = variable_on_cpu('b2', [self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            h2 = variable_on_cpu('h2', [self.hidden, self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), self.relu_clip)
            layer_2 = tf.nn.dropout(layer_2, self.keep_dropout)
    
        # 3rd layer
        with tf.name_scope('fc3'):
            b3 = variable_on_cpu('b3', [2 * self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            h3 = variable_on_cpu('h3', [self.hidden, 2 * self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), self.relu_clip)
            layer_3 = tf.compat.v1.nn.dropout(layer_3, self.keep_dropout)
    
        # 双向rnn
        with tf.name_scope('lstm'):
            # Forward direction cell:
            lstm_fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.cell_dim, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                        input_keep_prob=self.keep_dropout)
            # Backward direction cell:
            lstm_bw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.cell_dim, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                        input_keep_prob=self.keep_dropout)
    
            # `layer_3`  `[amax_stepsize, batch_size, 2 * cell_dim]`
            layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], 2 * self.cell_dim])
    
            outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                    cell_bw=lstm_bw_cell,
                                                                    inputs=layer_3,
                                                                    dtype=tf.float32,
                                                                    time_major=True,
                                                                    sequence_length=self.seq_length)
    
            # 连接正反向结果[amax_stepsize, batch_size, 2 * n_cell_dim]
            outputs = tf.concat(outputs, 2)
            # to a single tensor of shape [amax_stepsize * batch_size, 2 * n_cell_dim]
            outputs = tf.reshape(outputs, [-1, 2 * self.hidden])
    
        with tf.name_scope('fc5'):
            b5 = variable_on_cpu('b5', [self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            h5 = variable_on_cpu('h5', [(2 * self.hidden), self.hidden], tf.random_normal_initializer(stddev=self.stddev))
            layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), self.relu_clip)
            layer_5 = tf.nn.dropout(layer_5, self.keep_dropout)
    
        with tf.name_scope('fc6'):
            # 全连接层用于softmax分类
            b6 = variable_on_cpu('b6', [character], tf.random_normal_initializer(stddev=self.stddev))
            h6 = variable_on_cpu('h6', [self.hidden, character], tf.random_normal_initializer(stddev=self.stddev))
            layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    
        # 将2维[amax_stepsize * batch_size, character]转成3维 time-major [amax_stepsize, batch_size, character].        
        self.pred = tf.reshape(layer_6, [-1, batch_x_shape[0], character], name='pred')        
               
    #损失函数
    def loss_init(self):
        # 使用ctc loss计算损失
        self.loss = tf.reduce_mean(ctc_ops.ctc_loss(self.label, self.pred, self.seq_length))

    #优化器
    def optimizer_init(self):
        # 优化器        
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def accuracy_init(self):
        # 使用CTC decoder
        with tf.name_scope("decode"):
            self.decoded, _ = ctc_ops.ctc_beam_search_decoder(self.pred, self.seq_length, merge_repeated=False)
            
        # 计算编辑距离
        with tf.name_scope("accuracy"):
            distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.label)
            # 计算label error rate (accuracy)
            self.label_error_rate = tf.reduce_mean(distance, name='label_error_rate')


    def run(self, batch, source, source_lengths, sparse_labels, words, epoch):
        feed = {self.input: source, self.seq_length: source_lengths, self.label: sparse_labels, 
                    self.keep_dropout: self.keep_dropout_rate}

        # loss optimizer ;
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed)
        
        # 验证模型的准确率，比较耗时，我们训练的时候全力以赴，所以这里先不跑
        # if (batch + 1) % 1 == 0:            
        #     feed2 = {self.input: source, self.seq_length: source_lengths, self.label: sparse_labels, self.keep_dropout: 1.0}        
        #     decoded, label_error_rate = self.sess.run([self.decoded[0], self.label_error_rate], feed_dict=feed2)        
        #     dense_decodeds = tf.sparse_tensor_to_dense(decoded, default_value=0).eval(session=self.sess)
        #     dense_original_labels = sparse_tuple_to_text(sparse_labels, words)        
        #     counter = 0            
        #     print('Label err rate: ', label_error_rate)
        #     for dense_original_label, dense_decoded in zip(dense_original_labels, dense_decodeds):
        #         # convert to strings
        #         decoded_str = dense_to_text(dense_decoded, words)                 
        #         print('Original: {}'.format(dense_original_label))
        #         print('Decoded:  {}'.format(decoded_str))
        #         print('------------------------------------------')
        #         counter = counter + 1
                

        #每训练100次保存一下模型
        if (batch + 1) % 100 == 0:
            self.saver.save(self.sess, os.path.join(self.save_path + "birnn_speech_recognition.cpkt"), global_step=epoch)

        return loss


