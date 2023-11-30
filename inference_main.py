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
    parser.add_argument('-t', '--teacher', type=bool, default=False,help='if it is teacher model')

    parser.add_argument('--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('--clean_names', type=str, nargs='+', default=["A2爱0000.wav","A1打错了0000.wav","A1抱歉抱歉0000.wav","A5大鱼0015.wav","A6背对背拥抱0010.wav","A7没那么简单0033.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('--keys', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('--spk_list', type=str, nargs='+', default=['Alto-1'], help='合成目标说话人名称')
    parser.add_argument('--como_path', type=str, default="/logs/comom4gan/model_700000.pt", help='扩散模型路径')
    parser.add_argument('--como_config', type=str, default="/logs/comom4gan/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('--teacher_path', type=str, default="/logs/24k/diffusion/vocoder/m4gan/model_800000.pt", help='扩散模型路径')
    parser.add_argument('--teacher_config_path', type=str, default="/logs/24k/diffusion/vocoder/m4gan/config.yaml", help='扩散模型配置文件路径')

    args = parser.parse_args()

    clean_names = args.clean_names
    keys = args.keys
    spk_list = args.spk_list
    slice_db =-40 #默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50'
    wav_format = 'wav' # the format of the output audio
    pad_seconds = 0.5 # 推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现
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
                "slice_db" : slice_db,#-40
                "pad_seconds" : pad_seconds, # 0.5
                "clip_seconds" : clip, #0

            }
            audio = svc_model.slice_inference(**kwarg)
            step_num=args.diffusion_model_path.split('/')[-1].split('.')[0]
            if args.teacher:
                isdiffusion = "teacher"
            else:
                isdiffusion= "como"
            res_path = f'{resultfolder}/{clean_name}_{spk}_{isdiffusion}_{step_num}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
