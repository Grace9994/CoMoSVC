import glob
import os
from re import X
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
import logging
from multiprocessing import Pool
import sys
from glob import glob

def process(filename,wavformat,size):
    songname = filename.split('/')[-1].strip('.'+wavformat)#不包含.wav
    singer = filename.split('/')[-2]
    slice_name = './slice/'+singer+'/'+songname
    if not os.path.exists('./slice/'+singer):
        os.mkdir('./slice/'+singer)
    #去除静音部分
    audio_segment = AudioSegment.from_file(filename)#加载音频
    list_split_on_silence = split_on_silence(
        audio_segment, min_silence_len=600,
        silence_thresh=-40,                         
        keep_silence=400)
    sum=audio_segment[:1]  
    for i, chunk in enumerate(list_split_on_silence):
        sum=sum+chunk
        #sum.export(sum_name+"."+wavformat, format=wavformat)

    #进行切割片段
    #size = 10000  #切割的毫秒数
    chunks = make_chunks(sum, size)  

    for i, chunk in enumerate(chunks):
        chunk_name=slice_name+"_{0}.wav".format(i)       
        if not os.path.exists(chunk_name):
            #logger1.info(chunk_name)
            chunk.export(chunk_name, format="wav")

            

if __name__ == '__main__':
    wavformat = sys.argv[1]
    size = int(sys.argv[2])
    files=glob('./dataset_raw/*.'+wavformat)
    for file in files:
        process(file,wavformat,size)
    
   