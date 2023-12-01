import os
import torch
from torchaudio.transforms import Resample
from vocoder.m4gan.hifigan import HifiGanGenerator

from mel_processing import mel_spectrogram, MAX_WAV_VALUE
import utils

class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device # device
        self.vocodertype = vocoder_type
        if vocoder_type == 'm4-gan':
            self.vocoder = M4GAN(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate, keyshift=0):
        # resample
        if sample_rate == self.vocoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)# 这里是24k
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)

            audio_res = self.resample_kernel[key_str](audio)    #  对原始audio进行resample
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel

    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0]
        audio = self.vocoder(mel,f0)
        return audio


class M4GAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = utils.load_config(os.path.join(os.path.split(model_path)[0], 'config.yaml'))

    def sample_rate(self):
        return self.h.audio_sample_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.audio_num_mel_bins
        
    def extract(self, audio, keyshift=0):
        
        mel= mel_spectrogram(audio, self.h.fft_size, self.h.audio_num_mel_bins, self.h.audio_sample_rate, self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax, keyshift=keyshift).transpose(1,2)       
        # mel= mel_spectrogram(audio, 512, 80, 24000, 128, 512, 30, 12000, keyshift=keyshift).transpose(1,2) 
        return mel
    
    def load_checkpoint(self, filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    
    def forward(self, mel, f0):
        ckpt_dict = torch.load(self.model_path, map_location=self.device)
        state = ckpt_dict["state_dict"]["model_gen"]
        self.model = HifiGanGenerator(self.h).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.remove_weight_norm()
        self.model = self.model.eval()
        c = mel.transpose(2, 1)
        y = self.model(c,f0).view(-1)

        return y[None]
