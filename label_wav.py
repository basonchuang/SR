# encoding: utf-8
import os

# 获取文件夹下所有的WAV文件
def get_wavs(wav_path):
    wavs = []
    for (dirpath, _, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # print(filename)
                filename_path = os.path.join(dirpath, filename)
                # print(filename_path)
                wavs.append(filename_path)
    return wavs
 
 
# 获取wav文件对应的翻译文字
def get_tran_texts(wavs, tran_path):
    tran_texts = []
    for wav_file in wavs:
        (_, wav_filename) = os.path.split(wav_file)
        tran_file = os.path.join(tran_path, wav_filename + '.trn')
        # print(tran_file)
        if os.path.exists(tran_file) is False:
            return None

        with open(tran_file, encoding='utf-8') as fd:        
            text = fd.readline()
            tran_texts.append(text.split('\n')[0])
        
    return tran_texts
 
 
# 获取wav和对应的翻译文字
def get_wavs_and_tran_texts(wav_path, tran_path):
    wavs = get_wavs(wav_path)
    tran_texts = get_tran_texts(wavs, tran_path)
    #print(wavs[0], tran_texts[0])
    #print(len(wavs), len(tran_texts))
    return wavs, tran_texts


if __name__ == "__main__":
    wav_path = 'dataset/data_thchs30/train'
    tran_path = 'dataset/data_thchs30/data'
    wavs, labels = get_wavs_and_tran_texts(wav_path, tran_path)
    
