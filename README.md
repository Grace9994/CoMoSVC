# CoMoSVC

We proposed a consistency model based Singing Voice Conversion system, which is inspired by [CoMoSpeech](https://github.com/zhenye234/CoMoSpeech): One-Step Speech and Singing Voice Synthesis via Consistency Model. 

The paper and codebase of CoMoSVC are still being edited and will be completed as soon as possible.


# Environment
We have tested the code and it runs successfully on Python 3.8, so you can set up your Conda environment using the following command:

```shell
conda create -n Your_Conda_Environment_Name python=3.8
```
Then after activating your conda environment, you can install the required packages under it by:

```shell
pip install -r requirements.txt
```

## Dataset Preparation 

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

### 2. Split the training and validation datasets, and generate configuration files.

```shell
python preprocessing2_flist.py
```

##### diffusion.yaml



### 3. Generate features

You should first download https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view and then unzip the zip file by

```shell
unzip m4singer_hifigan.zip
```

the checkpoints of the vocoder will be in the `m4singer_hifigan` directory

Then you should download the checkpoint https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr and the put it in the `Content` directory to extract the content feature.

and then run the command

```shell
python preprocessing3_feature.py -c your_config_file -n num_processes 
```


## Training

### 1. train the teacher model

```shell
python train.py
```
The checkpoints will be saved in the `logs/teacher` directory

### 2. train the como model

#### if you want to adjust the config file, you can duplicate a new config file and modify some parameters.

You should download the pitch_extractor checkpoint from the  website https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view and then unzip the zip file by 

```shell
unzip m4singer_pe.zip
```


```shell
python train.py -t -c Your_new_configfile_path -p The_teacher_model_checkpoint_path 
```

## Inference
You should put the audios you want to convert under the `raw` directory firstly.

### Inference by teacher model

```shell
# Example
python inference_main.py -tm "logs/teacher/model_800000.pth" -tc "logs/teacher/config.yaml" -n "src.wav" -k 0 -s "target_singer"
```
-tm refers to the teacher_model_path

-tc refers to the teacher_config_path

-n refers to the source audio

-k refers to the pitch shift, it can be positive and negative (semitone) values

-s refers to the target singer

### Inference by student model

```shell
# Example
python inference_main.py -cm "logs/como/model_800000.pth" -tc "logs/como/config.yaml" -n "src.wav" -k 0 -s "target_singer" -t
```
-cm refers to the como_model_path
-cc refers to the como_config_path
