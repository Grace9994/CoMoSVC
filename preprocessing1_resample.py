import librosa
import os,sys,tqdm
import multiprocessing as mp
import soundfile as sf
from glob import glob

def resample_one(filename):
    fullname=filename.split('/')[-2]
    singer=fullname.split('_')[0]
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
    # /data/yiwenl/datasets/OpenSinger/ManRaw/12_凉凉/12_凉凉_1.wav
    fullname=input_path.split('/')[-2]
    singer=fullname.split('_')[0]
    out_dir = 'dataset/'+singer
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def resample_parallel(name,bin_idx,total_bins,num_process):
    input_paths = glob('dataset/'+name+'/*/*.wav')
    #input_paths = glob('/data/yiwenl/datasets/'+name+'/*/*.wav')    
    print("input_paths",len(input_paths))
    input_paths = input_paths[int(bin_idx)*len(input_paths)//int(total_bins):int(bin_idx+1)*len(input_paths)//int(total_bins)]
    # multiprocessing with progress bar
    pool = mp.Pool(num_process)
    for _ in tqdm.tqdm(pool.imap_unordered(resample_one, input_paths), total=len(input_paths)):
        pass

def path_parallel(name):
    input_paths = glob('dataset/'+name+'/*/*.wav')
    input_paths = list(set(input_paths))#sort
    print("input_paths",len(input_paths))
    for input_path in input_paths:
        mkdir_func(input_path)

if __name__ == "__main__":
    bin_idx = int(sys.argv[1])
    total_bins = int(sys.argv[2])
    num_process = int(sys.argv[3])
    name=sys.argv[4]
    path_parallel(name)
    resample_parallel(name, bin_idx,total_bins,num_process)
