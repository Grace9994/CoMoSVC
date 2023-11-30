## Dataset Preparation

Simply place the dataset in the `dataset_raw` directory with the following file structure:

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


## 🛠️ Preprocessing

### 0. Slice audio



**If you are using whisper-ppg encoder for training, the audio clips must shorter than 30s.**

### 1. Resample to 24000Hz and mono

```shell
python preprocessing1_resample.py
```

#### Cautions



```shell
python resample.py --skip_loudnorm
```

### 2. Automatically split the dataset into training and validation sets, and generate configuration files.

```shell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder has the following options

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
cnhubertlarge
dphubert
whisper-ppg-large
wavlmbase+
```

If the speech_encoder argument is omitted, the default value is `vec768l12`

**Use loudness embedding**

Add `--vol_aug` if you want to enable loudness embedding:

```shell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After enabling loudness embedding, the trained model will match the loudness of the input source; otherwise, it will match the loudness of the training set.

#### You can modify some parameters in the generated config.json and diffusion.yaml

* `keep_ckpts`: Keep the the the number of previous models during training. Set to `0` to keep them all. Default is `3`.

* `all_in_mem`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.
  
* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the GPU memory capacity.

* `vocoder_name`: Select a vocoder. The default is `nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.

* `duration`: The duration of the audio slicing during training, can be adjusted according to the size of the video memory, **Note: this value must be less than the minimum time of the audio in the training set!**

* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the video memory capacity.

* `timesteps`: The total number of steps in the diffusion model, which defaults to 1000.

* `k_step_max`: Training can only train `k_step_max` step diffusion to save training time, note that the value must be less than `timesteps`, 0 is to train the entire diffusion model, **Note: if you do not train the entire diffusion model will not be able to use only_diffusion!**

##### **List of Vocoders**

```
nsf-hifigan
nsf-snake-hifigan
```

### 3. Generate hubert and f0

```shell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor has the following options

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

If the training set is too noisy,it is recommended to use `crepe` to handle f0

If the f0_predictor parameter is omitted, the default value is `rmvpe`

If you want shallow diffusion (optional), you need to add the `--use_diff` parameter, for example:

```shell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

**Speed Up preprocess**

If your dataset is pretty large,you can increase the param `--num_processes` like that:

```shell
python preprocess_hubert_f0.py --f0_predictor dio --num_processes 8
```
All the worker will be assigned to different GPU if you have more than one GPUs.

After completing the above steps, the dataset directory will contain the preprocessed data, and the dataset_raw folder can be deleted.

## 🏋️‍ Training

### Sovits Model

```shell
python train.py -c configs/config.json -m 44k
```

### Diffusion Model (optional)

If the shallow diffusion function is needed, the diffusion model needs to be trained. The diffusion model training method is as follows:

```shell
python train_diff.py -c configs/diffusion.yaml
```

During training, the model files will be saved to `logs/44k`, and the diffusion model will be saved to `logs/44k/diffusion`

## 🤖 Inference

Use [inference_main.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/inference_main.py)

```shell
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"