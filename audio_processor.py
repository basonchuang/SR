# encoding: utf-8
import os
import numpy as np
from collections import Counter
from utils import get_mfcc_and_transcriptch, pad_sequences, sparse_tuple, get_labels_vector
from label_wav import get_wavs_and_tran_texts

charactersfile = "characters.txt"

class AudioProcessor(object):
    def __init__(self, wav_path, tran_path, features, contexts):        
        self.features = features
        self.contexts = contexts
        self.wavs, self.labels = get_wavs_and_tran_texts(wav_path, tran_path)
        # self.create_words_table()
        self.load_words_table()

    def batches_per_epoch(self, batch_size):
        return int(np.ceil(len(self.labels) / batch_size))

    def get_property(self):
        return self.words, self.words_size

    def load_words_table(self):
        self.words = []
        
        with open(charactersfile, 'r', encoding='utf-8') as fd:
            while True:
                c = fd.readline().replace('\n', '')
                if c:
                    self.words += [c]
                else:
                    break

        # print("words====>>>>", self.words)
        self.words_size = len(self.words)
        self.words_map = dict(zip(self.words, range(self.words_size)))    
        # print("words_map====>>>>", self.words_map)
        # # 将文字转成向量     
        # vector = get_labels_vector(self.words_map, self.labels[0])
        # print("labels[0]:", self.labels[0])
        # print("vector:", vector)

    def create_words_table(self):
        # 字表 
        all_words = []  
        for label in self.labels:  
            print(label)
            all_words += [word for word in label]
        
        #Counter，返回一个Counter对象集合，以元素为key，元素出现的个数为value
        counter = Counter(all_words)                
        #排序
        self.words = sorted(counter)
        
        self.words_size= len(self.words)

        with open(charactersfile, 'w', encoding='utf-8') as fd:
            for w in self.words:
                fd.write(w)
                fd.write('\n')

        self.words_map = dict(zip(self.words, range(self.words_size)))            
        #print("words_map====>>>>", self.words_map)
        #print("words====>>>>", self.words)    
        

    def next_batch(self, start_index=0, batch_size=1):
        filesize = len(self.labels)
        # 计算要获取的序列的开始和结束下标
        end_index = min(filesize, start_index + batch_size)
        index_list = range(start_index, end_index)
        # 获取要训练的音频文件路径和对于的译文
        labels = [self.labels[i] for i in index_list]
        wavs = [self.wavs[i] for i in index_list]
        # 将音频文件转成要训练的数据
        (source, _, target, _) = get_mfcc_and_transcriptch(wavs, labels, self.features,
                                                                self.contexts, self.words_map)
    
        start_index += batch_size
        # Verify that the start_index is not largVerify that the start_index is not ler than total available sample size
        if start_index >= filesize:
            start_index = -1
    
        # Pad input to max_time_step of this batch
        # 对齐处理，如果是多个文件，将长度统一，支持按最大截断或补0
        source, source_lengths = pad_sequences(source)
        # 返回序列的稀疏表示
        sparse_labels = sparse_tuple(target)
    
        return start_index, source, source_lengths, sparse_labels



if __name__ == "__main__":
    wav_path = 'dataset/data_thchs30/train'
    tran_path = 'dataset/data_thchs30/data'
 
    # processor = AudioProcessor(wav_path, tran_path, 0, 0)

    # 梅尔倒谱系数的个数
    features = 26
    # 对于每个时间序列，要包含上下文样本的个数
    contexts = 9
    # batch大小
    batch_size = 8

    processor = AudioProcessor(wav_path, tran_path, features, contexts)
    next_index = 0
   
    for batch in range(5):  # 一次batch_size，取多少次
        next_index, source, source_lengths, sparse_labels = processor.next_batch(next_index, batch_size)        
        print("source_lengths:", source_lengths)
        print("sparse_labels:", sparse_labels)
