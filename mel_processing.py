import math
import os
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, keyshift=0, speed=1,center=False):

    factor = 2 ** (keyshift / 12)       
    n_fft_new = int(np.round(n_fft * factor))
    win_size_new = int(np.round(win_size * factor))
    hop_length_new = int(np.round(hop_size * speed))


    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    mel_basis_key = str(fmax)+'_'+str(y.device)
    if mel_basis_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax) # 一个mel转换器，即这是一个函数，可以用来提取mel谱
        mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device) # 建 Mel 转换器，并将其转换为 PyTorch 的张量，并根据 y.device 将其放置在相应的设备上。
    
    keyshift_key = str(keyshift)+'_'+str(y.device)
    if keyshift_key not in hann_window:
        hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new- hop_length_new + 1) //2, win_size_new - y.size(-1) - pad_left)
    if pad_right < y.size(-1):
        mode = 'reflect'
    else:
        mode = 'constant'
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode = mode)
    y = y.squeeze(1)


    spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window[keyshift_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = F.pad(spec, (0, 0, 0, size-resize))
        spec = spec[:, :size, :] * win_size / win_size_new
    spec = torch.matmul(mel_basis[mel_basis_key], spec)

    spec = spectral_normalize_torch(spec)

    return spec


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    y_dtype = y.dtype
    if y.dtype == torch.bfloat16:
        y = y.to(torch.float32)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec).to(y_dtype)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


