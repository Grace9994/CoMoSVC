import librosa
import os,tqdm
import multiprocessing as mp
import soundfile as sf
from glob import glob
import argparse

def resample_one(filename):
    singer=filename.split('/')[-2]
    songname=filename.split('/')[-1]
    output_path='dataset/'+singer+'/'+songname
    if os.path.exists(output_path):
        return
    wav, sr = librosa.load(filename, sr=24000)
    # normalize the volume
    wav = wav / (0.00001+max(abs(wav)))*0.95
    # write to file using soundfile
    try:
        sf.write(output_path, wav, 24000)
    except:
        print("Error writing file",output_path)
        return

def mkdir_func(input_path):
    singer=input_path.split('/')[-2]
    out_dir = 'dataset/'+singer
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def resample_parallel(num_process):
    input_paths = glob('dataset_raw/*/*.wav')  
    print("input_paths",len(input_paths))
    # multiprocessing with progress bar
    pool = mp.Pool(num_process)
    for _ in tqdm.tqdm(pool.imap_unordered(resample_one, input_paths), total=len(input_paths)):
        pass

def path_parallel():
    input_paths = glob('dataset_raw/*/*.wav')
    input_paths = list(set(input_paths)) # sort
    print("input_paths",len(input_paths))
    for input_path in input_paths:
        mkdir_func(input_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num_process", type=int, default=5, help="the number of process")
    args = parser.parse_args()
    num_process = args.num_process
    path_parallel()
    resample_parallel(num_process)
