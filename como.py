import numpy as np
import torch
import copy
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
    def __init__(self, out_dims, n_layers, n_chans, n_hidden,teacher = True ):
        super().__init__()
        self.denoise_fn = WaveNet(out_dims, n_layers, n_chans, n_hidden)
        #self.denoise_fn = GradLogPEstimator2d(64) 
        self.teacher = teacher
        if not teacher: # 即非treacher
            self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
            self.denoise_fn_pretrained = copy.deepcopy(self.denoise_fn)

        self.P_mean =-1.2 # P_mean
        self.P_std =1.2# P_std
        self.sigma_data =0.5# sigma_data，即原始数据分布的标准差
 
        self.sigma_min= 0.002
        self.sigma_max= 80
        self.rho=7 
        self.N = 25         #100   
 
        
        # Time step discretization
        step_indices = torch.arange(self.N )   
        # 因为文中选取了sigma（t）=t，s（t）=1，所以sigma和t可以互相表示
        # t_step对应论文里（5），即sigma
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)])   # round_tensorj将数据转为tensor
 

    def EDMPrecond(self, x, sigma ,cond,denoise_fn):
        # 有条件输入，mask和cond是输入在F（theta）里的
        # 返回的是去除了噪声的结果，即clean signal
 
        sigma = sigma.reshape(-1, 1, 1 ) # 就是将sigma的形状变为[BATCHSIZE,1,1],inference的时候需要
        # Equ（7）里用到的系数
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
 

        F_x =  denoise_fn((c_in * x), c_noise.flatten(),cond) # [48, 187, 100]
        # print("F_x", F_x.shape)
        # print("x", x.shape)

        D_x = c_skip * x + c_out * (F_x .squeeze(1) ) # Equ(7)
        # D_x是去除了噪声的结果,对原始噪声的重构
        return D_x

    def EDMLoss(self, x_start, cond):
        # nonpadding 就是 mask

        #取sigma，用表格中的ln(sigma)服从一个正态分布
        rnd_normal = torch.randn([x_start.shape[0], 1,  1], device=x_start.device) # 生成形状为 [x_start.shape[0], 1,  1] 的服从标准高斯分布的随机噪声
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() # 括号里是ln(SIGMA),对ln(SIGMA)取exp

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 # 即lembda(sigma)，可以在表格中找到
        # print("x_start", x_start.shape)
        # print("cond", cond.shape)

 
        n = (torch.randn_like(x_start) ) * sigma # 先生成和x_start形状一致的标准高斯噪声，在噪声上加上condn是噪声，服从标准高斯，std
        D_yn = self.EDMPrecond(x_start + n, sigma ,cond,self.denoise_fn) # 去噪过后的结果
        loss = (weight * ((D_yn - x_start) ** 2)) # Equ(7)和Equ(8)之间的training loss
        loss=loss.unsqueeze(1).unsqueeze(1)
        loss=loss.mean() 
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
 
    # 对应EDM论文的Algorithm2，到第八步
    def edm_sampler(self,
         latents,  cond,
        num_steps=50, sigma_min=0.002, sigma_max=80, rho=7, 
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
        # S_churn=40 ,S_min=0.05,S_max=50,S_noise=1.003,# S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
        # S_churn=30 ,S_min=0.01,S_max=30,S_noise=1.007,
        # S_churn=30 ,S_min=0.01,S_max=1,S_noise=1.007,
        # S_churn=80 ,S_min=0.05,S_max=50,S_noise=1.003,
    ):
 

        # Time step discretization.
        step_indices = torch.arange(num_steps,  device=latents.device)#num_steps在como的foward里定义，是10，所以torch.Size([10])

        num_steps=num_steps + 1
        # t_step对应论文里（5），即每一步的sigma
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) #因为num_steps加了1，所以t_steps是torch.Size([11]),最后一维是0
        #t_steps其实也是每一步的sigma
        # Main sampling loop.
        x_next = latents * t_steps[0] #latents就是输入的纯高斯噪声，[cond[0],100,cond[1]]，
        x_next = x_next.transpose(1,2)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  
            #t_steps[:-1]是除了最后一个元素的所有元素，t_steps[1:]是除了第一个元素的所有元素，t_steps[1:]是除了第一个元素的所有元素
            x_cur = x_next
            # print('step',i+1)
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            # x_hat [1, 100, 465], t_hat [], cond [1, 465, 256]

            # Euler step.
            #x_hat=x_hat.transpose(1,2)
            #t_hat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            denoised = self.EDMPrecond(x_hat, t_hat, cond,self.denoise_fn) # 输入分别是带噪声的mel，sigma，cond
            d_cur = (x_hat - denoised) / t_hat # 7th step
            x_next = x_hat + (t_next - t_hat) * d_cur

        return x_next
  
    def CTLoss_D(self,y, cond): # 这里的y是gt_spec
        # y [48, 187, 100],cond [48, 187, 256]
        with torch.no_grad():
            mu = 0.95  
            for p, ema_p in zip(self.denoise_fn.parameters(), self.denoise_fn_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)
        n = torch.randint(1, self.N, (y.shape[0],))
        
        z = torch.randn_like(y) # 采样出和gt_spec形状一样的噪声
        #z = torch.randn_like(y)+ cond

        # c_t_d(n + 1 )：t_steps[i]
        tn_1 = self.c_t_d(n + 1).reshape(-1, 1, 1).to(y.device) # ([48, 1, 1]) t_n+1
        f_theta = self.EDMPrecond(y + tn_1 * z, tn_1, cond, self.denoise_fn)

        with torch.no_grad():
            tn = self.c_t_d(n ).reshape(-1, 1,   1).to(y.device)
            #euler step
            x_hat = y + tn_1 * z
            #  denoised = self.EDMPrecond(x_hat, t_hat, cond,self.denoise_fn) # 输入分别是带噪声的mel，sigma，cond
            denoised = self.EDMPrecond(x_hat, tn_1 , cond,self.denoise_fn_pretrained) 
            d_cur = (x_hat - denoised) / tn_1 # ODE Solver的结果
            y_tn = x_hat + (tn - tn_1) * d_cur
            f_theta_ema = self.EDMPrecond( y_tn, tn,cond, self.denoise_fn_ema)


 
        loss =   (f_theta - f_theta_ema.detach()) ** 2 # consistency model一般取 lembda=1,有了detach不会计算梯度
        loss=loss.unsqueeze(1).unsqueeze(1)
        loss=loss.mean() 
        
        return loss*1000

    def c_t_d(self, i ):
        return self.t_steps[i]

    def get_t_steps(self,N):
        N=N+1
        step_indices = torch.arange( N ) #, device=latents.device)
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N- 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        return  t_steps.flip(0)# 将t_step翻转

    def CT_sampler(self, latents, cond, t_steps=1):  
        # latents是生成的随机噪声

        if t_steps ==1:
            t_steps=[80]
        else:
            t_steps=self.get_t_steps(t_steps)

        t_steps = torch.as_tensor(t_steps).to(latents.device) # 将tensor放到对应的device上
        latents = latents * t_steps[0]
        latents = latents.transpose(1,2)
        # 对高斯噪声去噪后的结果-原始signal
        x = self.EDMPrecond(latents, t_steps[0],cond,self.denoise_fn)
        for t in t_steps[1:-1]: # t_steps中第2个到倒数第2个，即N-1到1
            z = torch.randn_like(x)
            x_tn = x +  (t ** 2 - self.sigma_min ** 2).sqrt()*z
            x = self.EDMPrecond(x_tn, t,cond,self.denoise_fn)
        return x

    def forward(self, x, cond, t_steps=50, infer=False):
        #这里的x是gt_spec,cond是所有特征拼接起来的向量
        
        if self.teacher: # teacher model  
            if not infer: # training
                loss = self.EDMLoss(x, cond)
                #loss = self.EDMLoss(x, cond,nonpadding) # y是真实mel，mu_y是特征
                # 对应的是diff_loss  = self.decoder(y, y_mask, mu_y)
                # nonpadding就是mask，             
                return loss
            else: # infer
                #这里的x是None，因为是从高斯噪声开始采样的
                shape = (cond.shape[0], 80, cond.shape[1])
                x = torch.randn(shape, device=cond.device)
                x=self.edm_sampler(x, cond,t_steps)# 随机噪声，特征，t_steps
            return x
        else:  #Consistency distillation
            if not infer: # training
                loss = self.CTLoss_D(x, cond)
                return loss
            else: # infer
                shape = (cond.shape[0], 80, cond.shape[1])
                x = torch.randn(shape, device=cond.device) # 并不是输入的gt，而是随机采样的噪声
                x=self.CT_sampler(x,cond,t_steps)
 
            return x
 
