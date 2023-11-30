## Dataset Preparation

Simply place the dataset in the `dataset_raw` directory with the following file structure:

```

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

### 0. Slice audio

```shell
python preprocessing0_slice.py wavformat slice_size
```

### 1. Resample to 24000Hz and mono

```shell
python preprocessing1_resample.py num_process
```



### 2. Split the training and validation datasets, and generate configuration files.

```shell
python preprocessing2_flist.py
```

##### diffusion.yaml



### 3. Generate features

```shell
python preprocessing3_features.py 
```


## Training

### 1. train the teacher model

```shell
python train.py
```


### 2. train the como model

#### if you want to adjust the config file, you can duplicate a new config file and modify some parameters.

```shell
python train.py -t False -c Your_new_configfile_path -p The_teacher_model_path 
```

## Inference

