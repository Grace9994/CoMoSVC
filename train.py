import argparse

import torch
from loguru import logger
from torch.optim import lr_scheduler

import os

from data_loaders import get_data_loaders
import utils
from solver import train
from ComoSVC import ComoSVC
from Vocoder import Vocoder
from utils import load_teacher_model_with_pitch
from utils import traverse_dir




def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='configs/diffusion.yaml',
        help="path to the config file")
    
    parser.add_argument(
        "-t",
        "--teacher",
        action='store_false',
        help="if it is the teacher model")

    parser.add_argument(
        "-s",
        "--total_steps",
        type=int,
        default=1,
        help="the number of iterative steps during inference")
    
    parser.add_argument(
        "-p",
        "--teacher_model_path",
        type=str,
        default="logs/teacher/model_800000.pt",
        help="path to teacher model")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    # load config
    args = utils.load_config(cmd.config)
    logger.info(' > config:'+ cmd.config)
    # teacher_or_not=cmd.teacher
    teacher_model_path=cmd.teacher_model_path
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    

    # load model
    if cmd.teacher:
        model = ComoSVC(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,#true
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    cmd.total_steps,
                    teacher=cmd.teacher
                    )
        
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.train.lr)
        initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    
    else:
        model = ComoSVC(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    cmd.total_steps,
                    teacher=cmd.teacher
                    )
        model = load_teacher_model_with_pitch(model,checkpoint_dir=cmd.teacher_model_path) # teacher model path


      #  optimizer = torch.optim.AdamW(params=model.decoder.denoise_fn.parameters())
        optimizer = torch.optim.AdamW(params=model.decoder.denoise_fn.parameters())
        path_pt = traverse_dir(args.env.comodir, ['pt'], is_ext=False)
        if len(path_pt)>0:
            initial_global_step, model, optimizer = utils.load_model(args.env.comodir, model, optimizer, device=args.device)
        else:
            initial_global_step = 0


    if cmd.teacher:
        logger.info(f' > The Teacher Model is training now.')
    else:
        logger.info(f' > The Student Model CoMoSVC is training now.')



    for param_group in optimizer.param_groups:
        if cmd.teacher:
            param_group['initial_lr'] = args.train.lr
            param_group['lr'] = args.train.lr * (args.train.gamma ** max(((initial_global_step-2)//args.train.decay_step),0) )
            param_group['weight_decay'] = args.train.weight_decay
        else:
            param_group['initial_lr'] = args.train.comolr
            param_group['lr'] = args.train.comolr * (args.train.gamma ** max(((initial_global_step-2)//args.train.decay_step),0) )
            param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,last_epoch=initial_global_step-2)
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, teacher=cmd.teacher)

    
