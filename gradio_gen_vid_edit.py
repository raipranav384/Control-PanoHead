from __future__ import annotations

# from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from torch import autocast,nn
import random
import os,sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'stable_diffusion'))
sys.path.append('./stable_diffusion')
sys.path.append(os.path.join(os.path.dirname(__file__), 'pose_3DDFA'))
from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
import argparse
from get_masks import get_masks
import datetime
from PIL import Image, ImageOps
from projector_withseg import run_projection
from gen_videos_proj_withseg import generate_images
from stable_diffusion.ldm.util import instantiate_from_config
from pose_3DDFA.pose_estimator import get_pose
import json
import math
import k_diffusion as K
from einops import rearrange
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip
parser=argparse.ArgumentParser()

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
parser.add_argument('--resolution',help="Resolution of edited img",default=512)

args=parser.parse_args()

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")
model_ip2p_ckpt="./models/instruct-pix2pix-00-22000.ckpt"
config_file="./configs/generate.yaml"
config = OmegaConf.load(config_file)
model_ip2p=load_model_from_config(config, model_ip2p_ckpt, None)


# null_token = model_ip2p.get_learned_conditioning([""])

def generate(
        input_image: Image.Image,
        instruction: str,
        steps_ip2p: int,
        steps: int,
        steps_pti: int,
        randomize_seed: bool,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        model_ip2p.eval().cuda()
        model_wrap = K.external.CompVisDenoiser(model_ip2p)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        # model_wrap
        # model_wrap_cfg.eval().cuda()
        null_token = model_ip2p.get_learned_conditioning([""])

        i=0
        if randomize_seed:
            seed = random.randint(0, 2 ** 32 - 1)
        # text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
        # image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale
        width, height = input_image.size
        factor = args.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        cur_time='tmp_data'

        # with torch.no_grad(), autocast("cuda"), model_ip2p.ema_scope():
        #     cond = {}
        #     cond["c_crossattn"] = [model_ip2p.get_learned_conditioning([instruction])]
        #     input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        #     input_image = rearrange(input_image, "h w c -> 1 c h w").to(model_ip2p.device)
        #     cond["c_concat"] = [model_ip2p.encode_first_stage(input_image).mode()]

        #     uncond = {}
        #     uncond["c_crossattn"] = [null_token]
        #     uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        #     sigmas = model_wrap.get_sigmas(steps_ip2p)

        #     extra_args = {
        #         "cond": cond,
        #         "uncond": uncond,
        #         "text_cfg_scale": text_cfg_scale,
        #         "image_cfg_scale": image_cfg_scale,
        #     }
        #     torch.manual_seed(seed)
        #     z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        #     z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        #     x = model_ip2p.decode_first_stage(z)
        #     x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        #     x = 255.0 * rearrange(x, "1 c h w -> h w c")
        #     edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        # model_ip2p.eval().cpu()
        # # model_wrap_cfg.eval().cpu()
        # del null_token
        # del model_wrap
        # del model_wrap_cfg
        # torch.cuda.empty_cache()
        # img=input_image.squeeze(0)
        # img_name='tmp'
        # img_dir=f'./data/{cur_time}/gen_img'
        # seg_mask_dir=f'./data/{cur_time}/seg_mask'
        # os.makedirs(img_dir,exist_ok=True)
        # os.makedirs(seg_mask_dir,exist_ok=True)
        # gen_img_pth=f'./data/{cur_time}/gen_img/{img_name}_{i:02d}.png'
        # seg_mask_pth=f'./data/{cur_time}/seg_mask/{img_name}_{i:02d}.png'
        # seg_mask=get_masks(img)
        # # img.save(gen_img_pth)
        # edited_image.save(gen_img_pth)
        # seg_mask=Image.fromarray(seg_mask)
        # if seg_mask.mode!='L':
        #     seg_mask=seg_mask.convert('L')
        # seg_mask.save(seg_mask_pth)
        # if seed == -1:
        #     seed = random.randint(0, 65535)
        # data_dic=get_pose(files=gen_img_pth)
        # with open(os.path.join(f"./data/{cur_time}/gen_img/",'dataset.json'),'w') as f:
        #     json.dump(data_dic,f)
        # print("POSE estimated!!")
        # run_projection(
        #     network_pkl=f'./models/{models}',
        #     target_img=img_dir,
        #     # target_img='./data/gen_img',
        #     target_seg=seg_mask_dir,
        #     # target_seg='./data/seg_mask',
        #     idx=0,
        #     outdir=f'./data/{cur_time}/projection',
        #     save_video=args.save_video,
        #     seed=seed,
        #     num_steps=steps,
        #     num_steps_pti=steps_pti,
        #     fps=args.fps,
        #     shapes=args.shapes,
        #     move2cpu=True,
        # )
        # torch.cuda.empty_cache()
        # os.makedirs(f'./data/{cur_time}/results',exist_ok=True)
        # generate_images(
        #     network_pkl=f'./data/{cur_time}/projection/{models}/0/fintuned_generator.pkl',
        #     latent=f'./data/{cur_time}/projection/{models}/0/projected_w.npz',
        #     output=f'./data/{cur_time}/results/out.mp4',
        #     truncation_psi=0.7,
        #     cfg='Head',
        #     shapes=True,
        #     move2cpu=True,
        # )
        torch.cuda.empty_cache()
        videoClip=VideoFileClip(f'./data/{cur_time}/results/out.mp4')
        vid_size=videoClip.size
        max_size=max(vid_size)
        factor=512/max_size            
        videClip=videoClip.resize(factor)
        print("Video_completed!!")
        videClip.write_videofile(f'./data/{cur_time}/results/out_resized.mp4')
        # videoClip.write_gif(f'./data/{cur_time}/results/out.gif')
        return [seed, f'./data/{cur_time}/results/out_resized.mp4',f'./data/{cur_time}/results/out.ply',]

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
                instruction = gr.Textbox(lines=1,Label="Edit Instruction",interactive=True)

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            # edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False)
            edited_video=gr.Video(label="Edited Video", type="mp4", interactive=False,height=512,width=512)
            input_image.style(height=512, width=512)
            # edited_image.style(height=512, width=512)

        with gr.Row():
            with gr.Column():
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
            with gr.Column():
                steps_ip2p = gr.Number(value=100, precision=0, label="Steps for IP2P", interactive=True)
                text_cfg_scale = gr.Number(value=7.5, precision=1, label="Text CFG Scale", interactive=True)
                image_cfg_scale = gr.Number(value=1.5, precision=1, label="Image CFG Scale", interactive=True)
        with gr.Row():
            save_button = gr.File(Label="Save ply", type='file', accept=None)
        gr.Markdown(help_text)
        generate_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps_ip2p,
                steps,
                steps_pti,
                randomize_seed,
                seed,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, edited_video,save_button],
        )
            
    demo.queue(concurrency_count=1)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
