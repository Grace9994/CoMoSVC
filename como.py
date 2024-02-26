import numpy as np
import torch
import copy
from pitch_extractor import PitchExtractor
from wavenet import WaveNet

import numpy as np
import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x


class Como(BaseModule):
    def __init__(self, out_dims, n_layers, n_chans, n_hidden,total_steps,teacher = True ):
        super().__init__()
        self.denoise_fn = WaveNet(out_dims, n_layers, n_chans, n_hidden)
        self.pe= PitchExtractor()
        self.teacher = teacher
        if not teacher: 
            self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
            self.denoise_fn_pretrained = copy.deepcopy(self.denoise_fn)

        self.P_mean =-1.2 
        self.P_std =1.2 
        self.sigma_data =0.5
 
        self.sigma_min= 0.002
        self.sigma_max= 80
        self.rho=7 
        self.N = 25   
        self.total_steps=total_steps
        self.spec_min=-6
        self.spec_max=1.5
        step_indices = torch.arange(self.N)   
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)])   # round_tensorj将数据转为tensor
    
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def EDMPrecond(self, x, sigma ,cond,denoise_fn):
        sigma = sigma.reshape(-1, 1, 1 ) 
        c_skip = self.sigma_data ** 2 / ((sigma-self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma-self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x =  denoise_fn((c_in * x), c_noise.flatten(),cond) 
        D_x = c_skip * x + c_out * (F_x .squeeze(1) ) 
        return D_x

    def EDMLoss(self, x_start, cond):
        rnd_normal = torch.randn([x_start.shape[0], 1,  1], device=x_start.device) 
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() 
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 
        n = (torch.randn_like(x_start) ) * sigma  # Generate Gaussian Noise
        D_yn = self.EDMPrecond(x_start + n, sigma ,cond,self.denoise_fn) # After Denoising
        loss = (weight * ((D_yn - x_start) ** 2)) 
        loss=loss.unsqueeze(1).unsqueeze(1)
        loss=loss.mean() 
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def edm_sampler(self,latents, cond,num_steps=50, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        # Time step discretization.
        step_indices = torch.arange(num_steps, device=latents.device)

        num_steps=num_steps + 1
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) 
        # Main sampling loop.
        x_next = latents * t_steps[0] 
        x_next = x_next.transpose(1,2)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  
            x_cur = x_next
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            denoised = self.EDMPrecond(x_hat, t_hat, cond, self.denoise_fn) # mel，sigma，cond
            d_cur = (x_hat - denoised) / t_hat # 7th step
            x_next = x_hat + (t_next - t_hat) * d_cur

        return x_next
  
    def CTLoss_D(self,y, cond): # y is the gt_spec
        with torch.no_grad():
            mu = 0.95  
            for p, ema_p in zip(self.denoise_fn.parameters(), self.denoise_fn_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)
        n = torch.randint(1, self.N, (y.shape[0],))
        
        z = torch.randn_like(y) # Gaussian Noise
        tn_1 = self.c_t_d(n + 1).reshape(-1, 1, 1).to(y.device) 
        f_theta = self.EDMPrecond(y + tn_1 * z, tn_1, cond, self.denoise_fn)

        with torch.no_grad():
            tn = self.c_t_d(n ).reshape(-1, 1,   1).to(y.device)
            #euler step
            x_hat = y + tn_1 * z
            denoised = self.EDMPrecond(x_hat, tn_1 , cond,self.denoise_fn_pretrained) 
            d_cur = (x_hat - denoised) / tn_1 
            y_tn = x_hat + (tn - tn_1) * d_cur
            f_theta_ema = self.EDMPrecond( y_tn, tn,cond, self.denoise_fn_ema)

        loss =   (f_theta - f_theta_ema.detach()) ** 2 # For consistency model, lembda=1
        loss=loss.unsqueeze(1).unsqueeze(1)
        loss=loss.mean()  
        
        return loss

    def c_t_d(self, i ):
        return self.t_steps[i]

    def get_t_steps(self,N):
        N=N+1
        step_indices = torch.arange( N ) #, device=latents.device)
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N- 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        return  t_steps.flip(0)# FLIP t_step

    def CT_sampler(self, latents, cond, t_steps=1):  
        if t_steps ==1:
            t_steps=[80]
        else:
            t_steps=self.get_t_steps(t_steps)
        t_steps = torch.as_tensor(t_steps).to(latents.device) 
        latents = latents * t_steps[0]
        latents = latents.transpose(1,2)
        x = self.EDMPrecond(latents, t_steps[0],cond,self.denoise_fn)
        for t in t_steps[1:-1]: # N-1 to 1
            z = torch.randn_like(x)
            x_tn = x +  (t ** 2 - self.sigma_min ** 2).sqrt()*z
            x = self.EDMPrecond(x_tn, t,cond,self.denoise_fn)
        return x

    def forward(self, x, cond, infer=False):

        if self.teacher: # teacher model  
            if not infer: # training
                x=self.norm_spec(x)
                loss = self.EDMLoss(x, cond)            
                return loss
            else: # infer
                shape = (cond.shape[0], 80, cond.shape[1])
                x = torch.randn(shape, device=cond.device)
                x=self.edm_sampler(x, cond, self.total_steps)
            return self.denorm_spec(x)
        else:  #Consistency distillation
            if not infer: # training
                x=self.norm_spec(x)
                loss = self.CTLoss_D(x, cond)
                return loss
            else: # infer
                shape = (cond.shape[0], 80, cond.shape[1])
                x = torch.randn(shape, device=cond.device) # The Input is the Random Noise
                x=self.CT_sampler(x,cond,self.total_steps) 
            return self.denorm_spec(x)

 
