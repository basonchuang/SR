# encoding: utf-8
import os
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from collections import Counter

from label_wav import get_wavs_and_tran_texts

# Constants
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
 
 
# 将稀疏矩阵的字向量转成文字
# tuple是sparse_tuple函数的返回值
def sparse_tuple_to_text_(tuple, words):
    # 索引
    indices = tuple[0]
    # 字向量
    values = tuple[1]
    
    dense = np.zeros(tuple[2]).astype(np.int32)

    for i in range(len(indices)):
        dense[indices[i][0]][indices[i][1]] = values[i]
    
    results = [''] * tuple[2][0]
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):            
            c = dense[i][j]
            c = ' ' if c == SPACE_INDEX else words[c]
            results[i] = results[i] + c
 
    return results

# 创建序列的稀疏表示，这个才是真的稀疏矩阵
def sparse_tuple_(sequences, dtype=np.int32):
    indices = []
    values = []
 
    for i, seq in enumerate(sequences):
        for j, value in enumerate(seq):
            if value != 0:  
                indices.extend([[i, j]])
                values.extend([value])
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape 
 

# 将密集矩阵的字向量转成文字
def dense_to_text(value, words):
    results = ''
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')
 
 
# 将稀疏矩阵的字向量转成文字，我们这里的稀疏矩阵也是假的
# tuple是sparse_tuple函数的返回值
def sparse_tuple_to_text(tuple, words):
    # 索引
    indices = tuple[0]
    # 字向量
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_INDEX else words[c]
        results[index] = results[index] + c
 
    return results

 
# 创建序列的稀疏表示，为了方便，我们这里只是做假的稀疏矩阵，我们只是需要稀疏矩阵的形式，因为ctc计算需要这种形式
def sparse_tuple(sequences, dtype=np.int32):
    indices = []
    values = []
 
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)
 
    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape

# 将字符转成向量，其实就是根据字找到字在words_map中所应对的下标
def get_labels_vector(words_map, txt_label=None):
    words_size = len(words_map)
 
    to_num = lambda word: words_map.get(word, words_size)

    labels_vector = list(map(to_num, txt_label))

    return labels_vector
 
#对齐处理
def pad_sequences(sequences, maxlen=None, dtype=np.float32, value=0.):
    #[478 512 503 406 481 509 422 465]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
 
    seqlen = len(sequences)
 
    #maxlen，该批次中，最长的序列长度
    if maxlen is None:
        maxlen = np.max(lengths)
 
    # 在下面的主循环中，从第一个非空序列中获取样本形状,以获取每个时序的mfcc特征数    
    sample_shape = tuple()
    for s in sequences:        
        if len(s) > 0:
            # (568, 494)
            sample_shape = np.asarray(s).shape[1:]
            break
    
    # (seqlen, maxlen, mfcclen)
    x = (np.ones((seqlen, maxlen) + sample_shape) * value).astype(dtype)
    
    for i, s in enumerate(sequences):
        if len(s) == 0:
            continue  # 序列为空，跳过
        
        if s.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (s.shape[1:], i, sample_shape))
 
        x[i, :len(s)] = s

    return x, lengths

# 将音频数据转为时间序列（列）和MFCC（行）的矩阵，将对应的译文转成字向量
def get_mfcc_and_transcriptch(wavs, labels, features, contexts, words_map):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
 
    for wav, label in zip(wavs, labels):
        # load audio and convert to features
        audio_data = audiofile_to_mfcc_vector(wav, features, contexts)
        audio_data = audio_data.astype('float32')
        # print(words_map)
        audio.append(audio_data)
        audio_len.append(len(audio_data))
 
        # load text transcription and convert to numerical array        
        target = get_labels_vector(words_map, label)  # txt_obj是labels
        # target = text_to_char_array(target)
        transcript.append(target)
        transcript_len.append(len(target))
 
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len

# 将音频信息转成MFCC特征
# 参数说明---audio_filename：音频文件   numcep：梅尔倒谱系数个数
#       numcontext：对于每个时间段，要包含的上下文样本个数
def audiofile_to_mfcc_vector(audio_filename, numcep, numcontext):
    # 加载音频文件
    fs, audio = wav.read(audio_filename)
    # 获取MFCC系数
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    # 打印MFCC系数的形状，得到比如(955, 26)的形状
    # 955表示时间序列，26表示每个序列的MFCC的特征值为26个
    # 这个形状因文件而异，不同文件可能有不同长度的时间序列，但是，每个序列的特征值数量都是一样的
    # print(np.shape(orig_inputs))
 
    # 因为我们使用双向循环神经网络来训练,它的输出包含正、反向的结
    # 果,相当于每一个时间序列都扩大了一倍,所以
    # 为了保证总时序不变,使用orig_inputs =
    # orig_inputs[::2]对orig_inputs每隔一行进行一次
    # 取样。这样被忽略的那个序列可以用后文中反向
    # RNN生成的输出来代替,维持了总的序列长度。
    orig_inputs = orig_inputs[::2]  # (478, 26)
    # print(np.shape(orig_inputs))
    # 因为我们讲解和实际使用的numcontext=9，所以下面的备注我都以numcontext=9来讲解
    # 这里装的就是我们要返回的数据，因为同时要考虑前9个和后9个时间序列，
    # 所以每个时间序列组合了19*26=494个MFCC特征数
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))
    # print(np.shape(train_inputs))#)(478, 494)
 
    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))
 
    # Prepare train_inputs with past and future contexts
    # time_slices保存的是时间切片，也就是有多少个时间序列
    time_slices = range(train_inputs.shape[0])
 
    # context_past_min和context_future_max用来计算哪些序列需要补零
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
 
    # 开始遍历所有序列
    for time_slice in time_slices:
        # 对前9个时间序列的MFCC特征补0，不需要补零的，则直接获取前9个时间序列的特征
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert (len(empty_source_past) + len(data_source_past) == numcontext)
 
        # 对后9个时间序列的MFCC特征补0，不需要补零的，则直接获取后9个时间序列的特征
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert (len(empty_source_future) + len(data_source_future) == numcontext)
 
        # 前9个时间序列的特征
        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past
 
        # 后9个时间序列的特征
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
 
        # 将前9个时间序列和当前时间序列以及后9个时间序列组合
        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)
 
        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert (len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)
 
    # 将数据使用正太分布标准化，减去均值然后再除以方差
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
 
    return train_inputs


def load_words_table_():
    words = []
    
    with open("characters.txt", 'r', encoding='utf-8') as fd:
        while True:
            c = fd.readline().replace('\n', '')
            if c:
                words += [c]
            else:
                break
    
    words_size = len(words)
    words_map = dict(zip(words, range(words_size)))    
    return words, words_map



if __name__ == "__main__":
    wav_path = 'dataset/data_thchs30/train'
    tran_path = 'dataset/data_thchs30/data'

    words, words_map = load_words_table_()    
    _, labels = get_wavs_and_tran_texts(wav_path, tran_path)

    ch_lable = get_labels_vector(words_map, labels[0])
    ch_lable1 = get_labels_vector(words_map, labels[1])
    # stuple = sparse_tuple_([ch_lable])    
    # stuple = sparse_tuple_([ch_lable, ch_lable1])
    stuple = sparse_tuple([ch_lable, ch_lable1])
    
    print("indices:", stuple[0])
    print("values:", stuple[1])
    print("shape:", stuple[2])
    # texts = sparse_tuple_to_text_(stuple, words)
    texts = sparse_tuple_to_text(stuple, words)
    print("texts:", texts)
    

