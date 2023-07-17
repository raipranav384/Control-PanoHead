from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--input',type=str,default='input_imgs')
parser.add_argument('-p','--prompt',type=str,help='Enter prompt')
parser.add_argument('--num_samples',type=int,default=1,help='No. of samples')
parser.add_argument('--image_resolution',default=512,type=int,help='Image Resolution')
parser.add_argument('--strength',type=float,default=1.0,help='Control Strength')
parser.add_argument('--guess_mode',action='store_true')
parser.add_argument('--detect_resolution',type=int,default=384)
parser.add_argument('--ddim_steps',type=int,default=20,help='Number of steps')
parser.add_argument('--scale',type=float,default=9.0)
parser.add_argument('-s','--seed',type=int,default=0)
parser.add_argument('--a_prompt',type=str,default='best quality, extremely detailed',help='Additional Prompt')
parser.add_argument('--n_prompt',type=str,default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',help='Negative Prompt')
parser.add_argument('--eta',type=float,default=0.0,help="eta (DDIM)")
input_image = gr.Image(source='upload', type="numpy")

args=parser.parse_args()


# apply_midas = MidasDetector()

model_name = 'control_v11p_sd15_normalbae'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/dreambooth_v15_2.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)

        # if det == 'None':
        detected_map = input_image.copy()
        # else:
        #     detected_map = preprocessor(resize_image(input_image, detect_resolution))
        #     detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
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

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

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
    imgs_pth=os.listdir(args.input)
    parent_pth=args.input
    for img_pth in imgs_pth:
        path=os.path.join(parent_pth,img_pth)
        print(path)
        img=cv2.imread(path)
        img_name=img_pth.split('.')[0]
        img=img[...,::-1]
        outputs=process(img,args.prompt,args.a_prompt,args.n_prompt,args.num_samples,args.image_resolution,args.detect_resolution,args.ddim_steps,args.guess_mode,args.strength,args.scale,args.seed,args.eta)
        for i,out in enumerate(outputs):
            cv2.imwrite(f'./results/{img_name}_{i:02d}.png',out[...,::-1])
    







