from __future__ import annotations

# from share import *
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
# from annotator.util import resize_image, HWC3
# from annotator.midas import MidasDetector
# from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler
import argparse
from get_masks import get_masks
import datetime
parser=argparse.ArgumentParser()
from PIL import Image
from projector_withseg import run_projection
from gen_videos_proj_withseg import generate_images

from pose_3DDFA.pose_estimator import get_pose
import json
import math

help_text = """
If you're not getting what you want, there may be a few reasons:
1. Is the image not changing enough? Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Incerasing the Text CFG weight, or
2. Conversely, is the image changing too much, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try:
    * Cropping the image so the face takes up a larger portion of the frame.
"""
parser.add_argument('--num-steps',              help='Number of optimization steps', type=int, default=700)
parser.add_argument('--num-steps-pti',          help='Number of optimization steps for pivot tuning', type=int, default=700)
# parser.add_argument('--seed',                   help='Random seed', type=int, default=666, show_default=True)
parser.add_argument('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True)
# parser.add_argument('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
parser.add_argument('--fps',                    help='Frames per second of final video', default=30)
parser.add_argument('--shapes', type=bool, help='Gen shapes for shape interpolation', default=True)

args=parser.parse_args()

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")
def generate(
        input_image: Image.Image,
        steps: int,
        steps_pti: int,
        randomize_seed: bool,
        seed: int,
    ):
        i=0
        if randomize_seed:
            seed = random.randint(0, 2 ** 32 - 1)
        cur_time='tmp_data'
        img=input_image
        img_name='tmp'
        img_dir=f'./data/{cur_time}/gen_img'
        seg_mask_dir=f'./data/{cur_time}/seg_mask'
        os.makedirs(img_dir,exist_ok=True)
        os.makedirs(seg_mask_dir,exist_ok=True)
        gen_img_pth=f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png'
        seg_mask_pth=f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png'
        seg_mask=get_masks(img)
        img.save(gen_img_pth)
        seg_mask=Image.fromarray(seg_mask)
        if seg_mask.mode!='L':
            seg_mask=seg_mask.convert('L')
        seg_mask.save(seg_mask_pth)
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
            num_steps=steps,
            num_steps_pti=steps_pti,
            fps=args.fps,
            move2cpu=True,
            shapes=args.shapes
        )
        os.makedirs(f'./data/{cur_time}/results',exist_ok=True)
        generate_images(
            network_pkl=f'./data/{cur_time}/projection/{models}/0/fintuned_generator.pkl',
            latent=f'./data/{cur_time}/projection/{models}/0/projected_w.npz',
            output=f'./data/{cur_time}/results/out.mp4',
            truncation_psi=0.7,
            cfg='Head',
            move2cpu=True,
            shapes=True
        )
        return [seed, f'./data/{cur_time}/results/out.mp4',f'./data/{cur_time}/results/out.ply',]

def main():
    def reset():
        return [0, "Randomize Seed", 1371, "Fix CFG", 7.5, 1.5, None]

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                generate_button = gr.Button("Generate")
            with gr.Column(scale=1, min_width=100):
                load_button = gr.Button("Load Example")
            with gr.Column(scale=1, min_width=100):
                reset_button = gr.Button("Reset")
            with gr.Column(scale=1, min_width=100):
                save_button = gr.File(Label="Save ply", type='file', accept=None)

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            # edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False)
            edited_video=gr.Video(label="Edited Video", type="mp4", interactive=False,height=512,width=512)
            input_image.style(height=512, width=512)
            # edited_image.style(height=512, width=512)

        with gr.Row():
            steps = gr.Number(value=700, precision=0, label="Steps", interactive=True)
            steps_pti=gr.Number(value=700, precision=0, label="Steps", interactive=True)
            randomize_seed = gr.Radio(
                ["Fix Seed", "Randomize Seed"],
                value="Randomize Seed",
                type="index",
                show_label=False,
                interactive=True,
            )
            seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)

        gr.Markdown(help_text)
        generate_button.click(
            fn=generate,
            inputs=[
                input_image,
                steps,
                steps_pti,
                randomize_seed,
                seed,
            ],
            outputs=[seed, edited_video,save_button],
        )
            
    demo.queue(concurrency_count=1)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
