from audio_processor import AudioProcessor
import tensorflow as tf
import time
import numpy as np
from birnn import BiRNN
import os

tf.compat.v1.disable_eager_execution()
# 梅尔倒谱系数的个数
features = 26
# 对于每个时间序列，要包含上下文样本的个数
contexts = 9
# batch大小
batch_size = 8


stddev = 0.046875

 
hidden = 1024
cell_dim = 1024

keep_dropout_rate = 0.95
relu_clip = 20

wav_path = 'dataset/data_thchs30/train'
tran_path = 'dataset/data_thchs30/data'

save_path = 'model/'

#迭代次数
epochs = 100

learning_rate = 0.00001

def main(argv=None):

    if not os.path.exists(wav_path) or not os.path.exists(tran_path):
        print('目录', wav_path, '或', tran_path, "不存在!")
        return

    processor = AudioProcessor(wav_path, tran_path, features, contexts)
    words, words_size = processor.get_property()

    birnn = BiRNN(features, contexts, batch_size, hidden, cell_dim, stddev, keep_dropout_rate, relu_clip, words_size+1, save_path, learning_rate)
        
    print('Run training epoch')
    start_epoch = birnn.get_property()
    

    for epoch in range(epochs):  # 样本集迭代次数
        epoch_start_time = time.time()
        if epoch < start_epoch:
            continue
 
        print("epoch start:", epoch, "total epochs= ", epochs)        
        batches_per_epoch = processor.batches_per_epoch(batch_size)
        print("total loop ", batches_per_epoch, "in one epoch，", batch_size, "items in one loop")

        next_index = 0        
        #######################run batch####
        for batch in range(batches_per_epoch):  # 一次batch_size，取多少次            
            next_index, source, source_lengths, sparse_labels = processor.next_batch(next_index, batch_size)
            batch_loss = birnn.run(batch, source, source_lengths, sparse_labels, words, epoch)

            epoch_duration = time.time() - epoch_start_time
 
            log = 'Epoch {}/{}, batch:{}, batch_loss: {:.3f}, time: {:.2f} sec'
            print(log.format(epoch, epochs, batch, batch_loss, epoch_duration))

if __name__ == '__main__':
    tf.compat.v1.app.run()
