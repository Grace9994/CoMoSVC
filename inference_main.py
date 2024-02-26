import logging
import soundfile
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import infer_tool
from infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='comosvc inference')
    parser.add_argument('-t', '--teacher', action="store_false",help='if it is teacher model')
    parser.add_argument('-ts', '--total_steps', type=int,default=1,help='the total number of iterative steps during inference')


    parser.add_argument('--clip', type=float, default=0, help='Slicing the audios which are to be converted')
    parser.add_argument('-n','--clean_names', type=str, nargs='+', default=['1.wav'], help='The audios to be converted,should be put in "raw" directory')
    parser.add_argument('-k','--keys', type=int, nargs='+', default=[0], help='To Adjust the Key')
    parser.add_argument('-s','--spk_list', type=str, nargs='+', default=['singer1'], help='The target singer')
    parser.add_argument('-cm','--como_model_path', type=str, default="./logs/como/model_800000.pt", help='the path to checkpoint of CoMoSVC')
    parser.add_argument('-cc','--como_config_path', type=str, default="./logs/como/config.yaml", help='the path to config file of CoMoSVC')
    parser.add_argument('-tm','--teacher_model_path', type=str, default="./logs/teacher/model_800000.pt", help='the path to checkpoint of Teacher Model')
    parser.add_argument('-tc','--teacher_config_path', type=str, default="./logs/teacher/config.yaml", help='the path to config file of Teacher Model')

    args = parser.parse_args()

    clean_names = args.clean_names
    keys = args.keys
    spk_list = args.spk_list
    slice_db =-40 
    wav_format = 'wav' # the format of the output audio
    pad_seconds = 0.5
    clip = args.clip

    if args.teacher:
        diffusion_model_path = args.teacher_model_path
        diffusion_config_path = args.teacher_config_path
        resultfolder='result_teacher'
    else:
        diffusion_model_path = args.como_model_path
        diffusion_config_path = args.como_config_path
        resultfolder='result_teacher'

    svc_model = Svc(diffusion_model_path,
                    diffusion_config_path,
                    args.total_steps,
                    args.teacher)
    
    infer_tool.mkdir(["raw", resultfolder])
    
    infer_tool.fill_a_to_b(keys, clean_names)
    for clean_name, tran in zip(clean_names, keys):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,# -40
                "pad_seconds" : pad_seconds, # 0.5
                "clip_seconds" : clip, #0

            }
            audio = svc_model.slice_inference(**kwarg)
            step_num=diffusion_model_path.split('/')[-1].split('.')[0]
            if args.teacher:
                isdiffusion = "teacher"
            else:
                isdiffusion= "como"
            res_path = f'{resultfolder}/{clean_name}_{spk}_{isdiffusion}_{step_num}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
