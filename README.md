<div align="center">
<h1>CoMoSVC: Consistency Model Based Singing Voice Conversion</h1>

[中文文档](./Readme_CN.md)
</div>

A consistency model based Singing Voice Conversion system is composed, which is inspired by [CoMoSpeech](https://github.com/zhenye234/CoMoSpeech): One-Step Speech and Singing Voice Synthesis via Consistency Model. 

This is an implemention of the paper [CoMoSVC](https://arxiv.org/pdf/2401.01792.pdf).
## Improvements
The subjective evaluations are illustrated through the table below.
<center><img src="https://comosvc.github.io/table3.jpg" width="800"></center>

## Environment
We have tested the code and it runs successfully on Python 3.8, so you can set up your Conda environment using the following command:

```shell
conda create -n Your_Conda_Environment_Name python=3.8
```
Then after activating your conda environment, you can install the required packages under it by:

```shell
pip install -r requirements.txt
```

## Download the Checkpoints
### 1. m4singer_hifigan

You should first download [m4singer_hifigan](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view) and then unzip the zip file by
```shell
unzip m4singer_hifigan.zip
```
The checkpoints of the vocoder will be in the `m4singer_hifigan` directory

### 2. ContentVec
You should download the checkpoint [ContentVec](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) and the put it in the `Content` directory to extract the content feature.

### 3. m4singer_pe
You should download the pitch_extractor checkpoint of the [m4singer_pe](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view) and then unzip the zip file by 

```shell
unzip m4singer_pe.zip
```

## Dataset Preparation 

You should first create the folders by

```shell
mkdir dataset_raw
mkdir dataset
```
You can refer to different preparation methods based on your needs.

Preparation With Slicing can help you remove the silent parts and slice the audio for stable training.


### 0. Preparation With Slicing

Please place your original dataset in the `dataset_slice` directory.

The original audios can be in any waveformat which should be specified in the command line. You can designate the length of slices you want, the unit of slice_size is milliseconds. The default wavformat and slice_size is mp3 and 10000 respectively.

```shell
python preparation_slice.py -w your_wavformat -s slice_size
```

### 1. Preparation Without Slicing

You can just place the dataset in the `dataset_raw` directory with the following file structure:

```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```


##  Preprocessing

### 1. Resample to 24000Hz and mono

```shell
python preprocessing1_resample.py -n num_process
```
num_process is the number of processes, the default num_process is 5.

### 2. Split the Training and Validation Datasets, and Generate Configuration Files.

```shell
python preprocessing2_flist.py
```


### 3. Generate Features

```shell
python preprocessing3_feature.py -c your_config_file -n num_processes 
```


## Training

### 1. Train the Teacher Model

```shell
python train.py
```
The checkpoints will be saved in the `logs/teacher` directory

### 2. Train the Consistency Model

If you want to adjust the config file, you can duplicate a new config file and modify some parameters.


```shell
python train.py -t -c Your_new_configfile_path -p The_teacher_model_checkpoint_path 
```

## Inference
You should put the audios you want to convert under the `raw` directory firstly.

### Inference by the Teacher Model

```shell
python inference_main.py -ts 50 -tm "logs/teacher/model_800000.pt" -tc "logs/teacher/config.yaml" -n "src.wav" -k 0 -s "target_singer"
```
-ts refers to the total number of iterative steps during inference for the teacher model

-tm refers to the teacher_model_path

-tc refers to the teacher_config_path

-n refers to the source audio

-k refers to the pitch shift, it can be positive and negative (semitone) values

-s refers to the target singer

### Inference by the Consistency Model

```shell
python inference_main.py -ts 1 -cm "logs/como/model_800000.pt" -cc "logs/como/config.yaml" -n "src.wav" -k 0 -s "target_singer" -t
```
-ts refers to the total number of iterative steps during inference for the student model

-cm refers to the como_model_path

-cc refers to the como_config_path

-t means it is not the teacher model and you don't need to specify anything after it 
