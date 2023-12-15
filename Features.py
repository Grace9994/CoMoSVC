import torch
import numpy as np
import pyworld
from fairseq import checkpoint_utils


class SpeechEncoder(object):
    def __init__(self, vec_path="Content/checkpoint_best_legacy_500.pt", device=None):
        self.model = None  # This is Model
        self.hidden_dim = 768
        pass


    def encoder(self, wav):
        """
        input: wav:[signal_length]
        output: embedding:[batchsize,hidden_dim,wav_frame]
        """
        pass



class ContentVec768L12(SpeechEncoder):
    def __init__(self, vec_path="Content/checkpoint_best_legacy_500.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)



class F0Predictor(object):
    def compute_f0(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        '''
        pass

    def compute_f0_uv(self,wav,p_len):
        '''
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        '''
        pass


class DioF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "dio"

    def interpolate_f0(self,f0):
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector

    def resize_f0(self,x, target_len):
        source = np.array(x)
        source[source<0.001] = np.nan
        target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
        res = np.nan_to_num(target)
        return res
        
    def compute_f0(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(self,wav,p_len=None):
        if p_len is None:
            p_len = wav.shape[0]//self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))
