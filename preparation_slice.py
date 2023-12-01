import glob
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
import argparse
from multiprocessing import Pool
from glob import glob

def process(filename,wavformat,size):
    songname = filename.split('/')[-1].strip('.'+wavformat)
    singer = filename.split('/')[-2]
    slice_name = './dataset_raw/'+singer+'/'+songname
    
    if not os.path.exists('./dataset_raw/'+singer):
        os.mkdir('./dataset_raw/'+singer)

    # Removing the silent parts
    audio_segment = AudioSegment.from_file(filename) # Loading the audio
    list_split_on_silence = split_on_silence(
        audio_segment, min_silence_len=600,
        silence_thresh=-40,                         
        keep_silence=400)
    sum=audio_segment[:1]  
    for i, chunk in enumerate(list_split_on_silence):
        sum=sum+chunk

    # Slicing
    chunks = make_chunks(sum, size)  

    for i, chunk in enumerate(chunks):
        chunk_name=slice_name+"_{0}.wav".format(i)       
        if not os.path.exists(chunk_name):
            #logger1.info(chunk_name)
            chunk.export(chunk_name, format="wav")

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w","--wavformat", type=str, default="wav", help="the wavformat of original data")
    parser.add_argument("-s","--size",  type=int, default=10000, help="the length of audio slices")
    args = parser.parse_args()
    wavformat = args.wavformat
    size = args.size
    files=glob('./dataset_slice/*/*.'+wavformat)

    for file in files:
        process(file,wavformat,size)
