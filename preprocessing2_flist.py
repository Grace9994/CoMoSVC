import argparse
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
            # The number of frames
            n_frames = wav_file.getnframes()
            # The sampling rate
            framerate = wav_file.getframerate()
            # The duration
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset", help="path to source dir")
    args = parser.parse_args()

    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0


    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
            if not file_name.endswith("wav"):
                continue
            if file_name.startswith("."):
                continue

            file_path = "/".join([args.source_dir, speaker, file_name])

            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            wavs.append(file_path)

        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing " + args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")


    d_config_template = utils.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id

    d_config_template["spk"] = spk_dict


    logger.info("Writing to configs/diffusion.yaml")
    utils.save_config("configs/diffusion.yaml",d_config_template)
