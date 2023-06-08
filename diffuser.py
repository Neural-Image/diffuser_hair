import argparse
import glob
import os
from PIL import Image, ImageFilter
import numpy as np
import torch
import facer

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, EulerAncestralDiscreteScheduler

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Input image folder path")
    parser.add_argument("--result_path", type=str, help="Result image folder path")
    args = parser.parse_args()
    return args

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def get_hair_mask(image_path, mask_blur = 4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    vis_seg_probs = seg_probs.argmax(dim=1)#/n_classes*255
    vis_seg_probs[vis_seg_probs != 10] = 0
    vis_seg_probs[vis_seg_probs == 10] = 1
    mask_img = vis_seg_probs.sum(0, keepdim=True)
    mask_img = vis_seg_probs.cpu().numpy()*255
    mask_img = mask_img.astype(np.uint8)
    mask_img = Image.fromarray(mask_img[0])
    mask_img = mask_img.convert("L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))
    return mask_img

def sd_controlnet_inpaint(init_image, mask_image):
    prompt = "(high detailed hair:1.2), RAW photo, a close up portrait photo, 8k uhd, dslr, soft \
    lighting, high quality, film grain, Fujifilm XT3"

    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, \
    drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low \
    quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, \
    mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, \
    blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
    disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
    extra arms, extra legs, fused fingers, too many fingers, long neck"

    cond_image = resize_for_condition_image(init_image, 1024)
    mask_image = mask_image.resize((1024, 1024))

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "XpucT/Deliberate", controlnet=controlnet, torch_dtype=torch.float16
    ).to('cuda')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, control_image=cond_image, image=init_image, \
                    strength = 1, mask_image=mask_image, \
                    num_inference_steps=20, guidance_scale=7, height=1024, width=1024, 
                    generator=torch.Generator(device="cuda").manual_seed(-1)).images[0]
    return image

def generate_final_result(image_path):
    init_image = Image.open(image_path).convert("RGB")
    mask_image = get_hair_mask(image_path)
    result_image = sd_controlnet_inpaint(init_image, mask_image)
    mask = np.array(mask_image.resize((1024, 1024)))/255.0
    mask = np.expand_dims(mask, axis=2)
    result = mask * np.array(result_image) + (1 - mask) * np.array(init_image.resize((1024,1024)))
    result = Image.fromarray(result.astype(np.uint8))
    return result_image

if __name__=="__main__":
    args = parse_agrs()
    image_files  = glob.glob(args.image_path + '/*.png')
    os.makedirs(args.result_path, exist_ok=True)
    for image_file in image_files:
        print(image_file)
        result =  generate_final_result(image_file)
        file_name = os.path.basename(image_file)
        result_file_name = os.path.join(args.result_path, file_name)
        result.save(result_file_name)