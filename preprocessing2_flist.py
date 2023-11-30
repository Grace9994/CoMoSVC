import argparse
import json
import os
import re
import wave
from random import shuffle

from loguru import logger
from tqdm import tqdm

import utils

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="/data/yiwenl/comosvc/so-vits-svc/dataset/24k/OpenSinger", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    parser.add_argument("--data", type=str, help="Define the dataset",required=True)
    args = parser.parse_args()
    
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0

    train_list='filelists/'+args.data+'/train.txt'
    val_list='filelists/'+args.data+'/val.txt'
    source_dir ='dataset/'+args.data

    for speaker in tqdm(os.listdir(source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        for file_name in os.listdir(os.path.join(source_dir, speaker)):
            if not file_name.endswith("wav"):
                continue
            if file_name.startswith("."):
                continue

            file_path = "/".join([source_dir, speaker, file_name])

            if not pattern.match(file_name):
                logger.warning("Detected non-ASCII file name: " + file_path)

            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            wavs.append(file_path)

        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + train_list)
    with open(train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing " + val_list)
    with open(val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")


    d_config_template = utils.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id

    d_config_template["spk"] = spk_dict
    
    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id

    
    config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 768
    d_config_template["data"]["encoder_out_channels"] = 768

    if args.vol_aug:
        config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True

    if args.tiny:
        config_template["model"]["filter_channels"] = 512

    logger.info("Writing to configs/config.json")
    with open("configs/"+args.data+"/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    logger.info("Writing to configs/diffusion.yaml")
    utils.save_config("configs/"+args.data+"/diffusion.yaml",d_config_template)
