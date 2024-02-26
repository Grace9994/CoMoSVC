<div align="center">
<h1>CoMoSVC: One-Step Consistency Model Based Singing Voice Conversion</h1>
</div>

基于一致性模型的歌声转换及克隆系统，可以一步diffusion采样进行歌声转换，是对论文[CoMoSVC](https://arxiv.org/pdf/2401.01792.pdf)的实现。工作基于[CoMoSpeech](https://github.com/zhenye234/CoMoSpeech): One-Step Speech and Singing Voice Synthesis via Consistency Model. 



# 环境配置
Python 3.8环境下创建Conda虚拟环境:

```shell
conda create -n Your_Conda_Environment_Name python=3.8
```
安装相关依赖库:

```shell
pip install -r requirements.txt
```
## 下载checkpoints
### 1. m4singer_hifigan
下载vocoder [m4singer_hifigan](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view) 并解压

```shell
unzip m4singer_hifigan.zip
```

vocoder的checkoint将在`m4singer_hifigan`目录中

### 2. ContentVec

下载 [ContentVec](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 放置在`Content`路径，以提取歌词内容特征。

### 3. m4singer_pe

下载pitch_extractor [m4singer_pe](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view) ,并解压到根目录

```shell
unzip m4singer_pe.zip
```


## 数据准备 


构造两个空文件夹

```shell
mkdir dataset_raw
mkdir dataset
```

请自行准备歌手的清唱录音数据，随后按照如下操作。

### 0. 带切片的数据准备流程

请将你的原始数据集放在 dataset_slice 目录下。

原始音频可以是任何波形格式，应在命令行中指定。你可以指定你想要的切片长度，切片大小的单位是毫秒。默认的文件格式和切片大小分别是mp3和10000。

```shell
python preparation_slice.py -w 你的文件格式 -s 切片大小
```

### 1. 不带切片的数据准备流程

你可以只将数据集放在 `dataset_raw` 目录下，按照以下文件结构：


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


##  预处理

### 1. 重采样为24000Hz和单声道

```shell
python preprocessing1_resample.py -n num_process
```
num_process 是进程数，默认5。


### 2. 分割训练和验证数据集，并生成配置文件

```shell
python preprocessing2_flist.py
```


### 3.  生成特征




执行下面代码以提取所有特征

```shell
python preprocessing3_feature.py -c your_config_file -n num_processes 
```




## 训练

### 1. 训练 teacher model

```shell
python train.py
```
Checkpoint将存于 `logs/teacher` 目录中

### 2. 训练 consistency model

#### 如果你想调整配置文件，你可以复制一个新的配置文件并修改一些参数。



```shell
python train.py -t -c Your_new_configfile_path -p The_teacher_model_checkpoint_path 
```

## 推理

你应该首先将你想要转换的音频放在`raw`目录下。

### 采用教师模型的推理

```shell
python inference_main.py -ts 50 -tm "logs/teacher/model_800000.pt" -tc "logs/teacher/config.yaml" -n "src.wav" -k 0 -s "target_singer"
```
-ts 教师模型推理时的迭代步数

-tm 教师模型路径

-tc 教师模型配置文件

-n source音频路径

-k pitch shift，可以是正负semitone值

-s 目标歌手



### 采用CoMoSVC进行推理

```shell
python inference_main.py -ts 1 -cm "logs/como/model_800000.pt" -cc "logs/como/config.yaml" -n "src.wav" -k 0 -s "target_singer" -t
```
-ts 学生模型推理时的迭代步数

-cm CoMoSVC模型路径

-cc CoMoSVC模型配置文件

-t 加上该参数并保留后续为空代表不是教师模型
