from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'pose_3DDFA'))
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse
from get_masks import get_masks
import datetime
parser=argparse.ArgumentParser()
from PIL import Image
from projector_withseg import run_projection
from pose_3DDFA.pose_estimator import get_pose
import json
parser.add_argument('--input',type=str,default='./data/input_mask')
parser.add_argument('-p','--prompt',type=str,help='Enter prompt')
parser.add_argument('--num_samples',type=int,default=1,help='No. of samples')
parser.add_argument('--image_resolution',default=512,type=int,help='Image Resolution')
parser.add_argument('--strength',type=float,default=1,help='Control Strength')
parser.add_argument('--guess_mode',action='store_true')
parser.add_argument('--detect_resolution',type=int,default=384)
parser.add_argument('--ddim_steps',type=int,default=20,help='Number of steps')
parser.add_argument('--scale',type=float,default=9.0)
parser.add_argument('-s','--seed',type=int,default=-1)
parser.add_argument('--a_prompt',type=str,default='best quality, extremely detailed',help='Additional Prompt')
parser.add_argument('--n_prompt',type=str,default='longbody, lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality',help='Negative Prompt') # Removed missing fingers for face segment
parser.add_argument('--eta',type=float,default=0.0,help="eta (DDIM)")

# parser.add_argument('--network', help='Network pickle filename', required=True)
# @click.option('--target', 'target_fname',       help='Target image file to project to', required=True, metavar='FILE|DIR')
# parser.add_argument('--target_img', 'target_img',       help='Target image folder', required=True, metavar='FILE|DIR')
# parser.add_argument('--target_seg', 'target_seg',       help='Target segmentation folder', required=True, metavar='FILE|DIR')
# parser.add_argument('--idx',                    help='index from dataset', type=int, default=0,  metavar='FILE|DIR')
parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=700)
parser.add_argument('--num-steps-pti',          help='Number of optimization steps for pivot tuning', type=int, default=700)
# parser.add_argument('--seed',                   help='Random seed', type=int, default=666, show_default=True)
parser.add_argument('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True)
# parser.add_argument('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
parser.add_argument('--fps',                    help='Frames per second of final video', default=30)
parser.add_argument('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False)


input_image = gr.Image(source='upload', type="numpy")

args=parser.parse_args()

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")
# apply_midas = MidasDetector()

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./models/face_segment_sd_21.ckpt', location='cuda'),strict=False)
# model.load_state_dict(load_state_dict('./models/yogi.pth', location='cuda'),strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        # _, detected_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
        # detected_map = HWC3(detected_map)
        detected_map=resize_image(input_image, detect_resolution)
        # cv2.imwrite('normal_map.png',detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


if os.path.isdir(args.input):
    cur_time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    imgs_pth=os.listdir(args.input)
    parent_pth=args.input
    for img_pth in imgs_pth:
        path=os.path.join(parent_pth,img_pth)
        print(path)
        img=cv2.imread(path)
        img_name=img_pth.split('.')[0]
        img=img[...,::-1]
        outputs=process(img,args.prompt,args.a_prompt,args.n_prompt,args.num_samples,args.image_resolution,args.detect_resolution,args.ddim_steps,args.guess_mode,args.strength,args.scale,args.seed,args.eta)
        img_dir=f'./data/{cur_time}/gen_img'
        seg_mask_dir=f'./data/{cur_time}/seg_mask'
        os.makedirs(img_dir,exist_ok=True)
        os.makedirs(seg_mask_dir,exist_ok=True)
        for i,out in enumerate(outputs):
            gen_img_pth=f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png'
            seg_mask_pth=f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png'
            out_img=Image.fromarray(out)
            seg_mask=get_masks(out_img)
            # cv2.imwrite(f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png',out[...,::-1])
            # cv2.imwrite(f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png',out[...,::-1])
            out_img.save(gen_img_pth)
            seg_mask=Image.fromarray(seg_mask)
            if seg_mask.mode!='L':
                seg_mask=seg_mask.convert('L')
            seg_mask.save(seg_mask_pth)
            seed=args.seed
            if seed == -1:
                seed = random.randint(0, 65535)
            data_dic=get_pose(files=gen_img_pth)
            with open(os.path.join(f"./data/{cur_time}/gen_img/",'dataset.json'),'w') as f:
                json.dump(data_dic,f)
            print("POSE estimated!!")
            run_projection(
                network_pkl=f'./models/{models}',
                target_img=img_dir,
                # target_img='./data/gen_img',
                target_seg=seg_mask_dir,
                # target_seg='./data/seg_mask',
                idx=i,
                outdir=f'./data/{cur_time}/projection',
                save_video=args.save_video,
                seed=seed,
                num_steps=args.num_steps,
                num_steps_pti=args.num_steps_pti,
                fps=args.fps,
                shapes=args.shapes
            )
