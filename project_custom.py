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
from gen_videos_proj_withseg import generate_images

from pose_3DDFA.pose_estimator import get_pose
import json
parser.add_argument('-i','--input',type=str,default='./data/input_mask')
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


cur_time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
imgs_pth=os.listdir(args.input)
parent_pth=args.input
for img_pth in imgs_pth:
    i=0
    path=os.path.join(parent_pth,img_pth)
    print(path)
    img=cv2.imread(path)
    img_name=img_pth.split('.')[0]
    img=img[...,::-1]
    img_dir=f'./data/{cur_time}/gen_img'
    seg_mask_dir=f'./data/{cur_time}/seg_mask'
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(seg_mask_dir,exist_ok=True)

    gen_img_pth=f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png'
    seg_mask_pth=f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png'
    img=Image.fromarray(img)
    seg_mask=get_masks(img)
    # cv2.imwrite(f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png',out[...,::-1])
    # cv2.imwrite(f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png',out[...,::-1])
    img.save(gen_img_pth)
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
        idx=0,
        outdir=f'./data/{cur_time}/projection',
        save_video=args.save_video,
        seed=seed,
        num_steps=args.num_steps,
        num_steps_pti=args.num_steps_pti,
        fps=args.fps,
        shapes=args.shapes
    )
    # cur_time='2023-07-01_00-51-45'
    os.makedirs(f'./data/{cur_time}/results',exist_ok=True)
    generate_images(
        network_pkl=f'./data/{cur_time}/projection/{models}/0/fintuned_generator.pkl',
        latent=f'./data/{cur_time}/projection/{models}/0/projected_w.npz',
        output=f'./data/{cur_time}/results/out.mp4',
        truncation_psi=0.7,
        cfg='Head',
        shapes=True
    )