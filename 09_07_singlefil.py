import argparse
import glob
import os
import PIL
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import random
import torch
import facer
from pathlib import Path
import tempfile
import gradio as gr
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline

from MODCDGnet_8_23_new import mod_cdgnet

import json

# Load the JSON data into a Python dictionary
#with open('1prompt.json', 'r') as file:
#with open('1image_8_27.json', 'r') as file:
#with open('1image_9_8.json', 'r') as file:
with open('10prompt_9_15.json', 'r') as file:
    data = json.load(file)

gender = 'm'  # or 'm'

position_third = None

resolution = 896

global w_face, h_face, x_face, y_face


def get_prompts(category, gender):
    prompt = data[category].get(f'prompt_{gender}', None)
    neg_prompt = data[category].get(f'neg_prompt_{gender}', None)

    if prompt == "None":
        prompt = None
    
    return prompt, neg_prompt


def set_random_third_position():
    global position_third
    # Calculate a 5% range for 1/3 and 2/3 positions
    left_third_range = (7/18+0.05, 7/18 + 0.06)
    right_third_range = (11/18 - 0.06, 11/18-0.05)

    # Choose one of the ranges randomly and then get a random value within that range
    chosen_range = random.choice([left_third_range, right_third_range])
    position_third = random.uniform(chosen_range[0], chosen_range[1])

def erode_mask(mask, kernel_size):
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask

def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))
    
    # Adjust for horizontal concatenation
    total_width = sum(widths)
    total_height = max(heights)
    
    new_img = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size

    # Pad the image to have a 1:1 aspect ratio
    if H > W:
        diff = H - W
        padding = (diff // 2, 0, diff - (diff // 2), 0)
    else:
        diff = W - H
        padding = (0, diff // 2, 0, diff - (diff // 2))
    input_image = ImageOps.expand(input_image, border=padding, fill='black')

    k = float(resolution) / max(H, W)
    H = W = int(round(H * k / 64.0)) * 64

    if H != resolution:
        H = resolution
    if W != resolution:
        W = resolution

    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def resize_composition(input_image: Image, resolution):
    global h_face, x_face, y_face, w_face
    
    if position_third is None:
        set_random_third_position()

    W, H = input_image.size
    print(f"x_face, y_face, w_face, h_face is {x_face}, {y_face}, {w_face}, {h_face}")

    # Target vertical position for the bounding box
    target_top = H * 0.25
    target_bottom = H * 0.6

    # Scaling factor to fit the bounding box to the target height
    target_height = target_bottom - target_top
    scale_factor = target_height / h_face

    # Calculate new image dimensions
    new_W = int(W * scale_factor)
    new_H = int(H * scale_factor)

    # Scale the original image
    scaled_image = input_image.resize((new_W, new_H), resample=Image.LANCZOS)

    # Calculate the padding to place the bounding box at the target vertical position
    padding_top = int(target_top - y_face * scale_factor)

    # Calculate the target horizontal position for the bounding box center 
    target_center_x = W * position_third
    bounding_box_center_x_scaled = (x_face + (w_face / 2)) * scale_factor
    padding_left = int(target_center_x - bounding_box_center_x_scaled)

    # Create a new blank white image with original dimensions
    new_image = Image.new('RGB', (W, H), 'white')

    # Paste the scaled image at the correct position on the white canvas
    new_image.paste(scaled_image, (padding_left, padding_top))

    # Crop 12.5% off the left and right sides
    crop_width = int(W * 0.125)
    new_image = new_image.crop((crop_width, 0, W - crop_width, H))

    return new_image


def resize_for_init_image(input_image: Image):
    W, H = input_image.size
    global h_face, x_face, y_face, w_face
    if position_third is None:
        set_random_third_position()

    # Target vertical position for the bounding box
    target_top = H * 0.25
    target_bottom = H * 0.6

    # Scaling factor to fit the bounding box to the target height
    target_height = target_bottom - target_top
    scale_factor = target_height / h_face

    # Calculate new image dimensions
    new_W = int(W * scale_factor)
    new_H = int(H * scale_factor)

    # Scale the original image
    scaled_image = input_image.resize((new_W, new_H), resample=Image.LANCZOS)

    # Calculate the padding to place the bounding box at the target vertical position
    padding_top = int(target_top - y_face * scale_factor)

    # Calculate the target horizontal position for the bounding box center 
    # (positioning it on the first third line)
    target_center_x = W * position_third
    
    # Calculate the padding to place the bounding box at the target horizontal position
    bounding_box_center_x_scaled = (x_face + (w_face / 2)) * scale_factor
    padding_left = int(target_center_x - bounding_box_center_x_scaled)

    # Create a new blank image with original dimensions
    new_image = Image.new('RGB', (W, H), 'black')

    # Paste the scaled image at the correct position
    new_image.paste(scaled_image, (padding_left, padding_top))

    # Crop 12.5% off the left and right sides
    crop_width = int(W * 0.125)
    new_image = new_image.crop((crop_width, 0, W - crop_width, H))

    return new_image


def get_head_mask(image_path, mask_blur=40, include_hair=True, crop=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    ### adjust kernel and blur size based on face size  ## '
    f_box = faces['rects'].cpu()[0]
    #print(f"f_box is {f_box}")

    global x_face, y_face, w_face, h_face
    x_face = int(f_box[0])
    y_face = int(f_box[1]) 
    w_face = int(f_box[2] - f_box[0])
    h_face = int(f_box[3] - f_box[1])
    #print(f"w_face is {w_face}, h_face is {h_face}")

    # print('image.shape', image.shape)
    _, _, h, w = image.shape
    side = max(h, w)

    face_ratio = min(w_face / side, h_face / side)
    #print(f"face_ratio: {face_ratio}")

    if face_ratio < 0.10:
        factor_1 = 4
    if .10 <= face_ratio < 0.2:
        factor_1 = 5
    if 0.2 <= face_ratio < 0.3:
        factor_1 = 8
    if 0.3 <= face_ratio < 0.4:
        factor_1 = 9
    if 0.4 <= face_ratio < 0.6:
        factor_1 = 11
    if face_ratio >= 0.6:
        factor_1 = 13

    print(f'face ratio is {factor_1}')


    hair_mask, face_mask, body_mask = mod_cdgnet(image_path)
    mask_img = resize_for_condition_image(face_mask, resolution)

    # Convert the PIL Image back to a numpy array
    mask_img = np.array(mask_img)

    #print(f"erosion is {int(factor_1/1.5)}, blur is {factor_1//2}")

    mask_img = erode_mask(mask_img, kernel_size=factor_1)

    # After dilation, convert back to PIL Image for blurring
    mask_img = Image.fromarray(mask_img)

    #mask_img = mask_img.filter(ImageFilter.GaussianBlur(factor_1//2))
    mask_img = PIL.ImageOps.invert(mask_img)
    
    return mask_img


def sd_inpaint(init_image, mask_image, prompt, negative_prompt):

    #dreamshaper_8Inpainting.safetensors
    #realisticVisionV51_v51VAE-inpainting.safetensors
    pipe = StableDiffusionInpaintPipeline.from_single_file("/home/heran/realistic-vision-inpaint/realisticVisionV51_v51VAE-inpainting.safetensors", torch_dtype=torch.float16, use_safetensors=True).to('cuda')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    #pipe.load_textual_inversion("/home/heran/DreamShaper/BadDream.pt", token="BadDream")
    #pipe.load_textual_inversion("/home/heran/DreamShaper/UnrealisticDream.pt", token="UnrealisticDream")
    #pipe.unet.load_attn_procs("/home/heran/realistic-vision-inpaint/koreanDollLikeness.safetensors")

    pipe.load_lora_weights("/home/heran/realistic-vision-inpaint", weight_name="koreanDollLikeness.safetensors")

    # Disables safety checksimac 
    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
    pipe.safety_checker = disabled_safety_checker

    seed = random.randint(0, 2 ** 32 - 1)

    images = pipe(prompt=prompt, negative_prompt=negative_prompt,
         image=init_image, strength=1, mask_image=mask_image, \
         num_inference_steps=20, guidance_scale=7.5, height=896, width=672, num_images_per_prompt=1, cross_attention_kwargs={"scale": 0.5},
         generator=torch.Generator(device="cuda").manual_seed(seed)).images

    return images



def generate_images(prompts, negative_prompt, input_path, output_path, change_hair):
    # Split the prompts by line
    prompts = prompts.split('\n')

    # Get list of all image files in the input path
    image_files = glob.glob(os.path.join(input_path, "*"))

    # Process each image
    for image_path in image_files:
        #print(f"The image_path is {image_path}")
        image_grid = []

        # Generate result image
        init_image = Image.open(image_path).convert("RGB")
        ori_init_image = resize_for_condition_image(init_image, resolution)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            ori_init_image.save(temp.name)
            temp.close()  # Close the file so it can be opened by another process

            ori_mask_image = get_head_mask(temp.name, include_hair=change_hair, crop=True)

            if (w_face > 200 and h_face > 200):           

                for category in data.keys():
                    prompt, negative_prompt = get_prompts(category, gender)

                    print(prompt)
                    print(negative_prompt)

                    if prompt is not None:

                        set_random_third_position()

                        init_image = resize_for_init_image(ori_init_image)
                        mask_image = resize_composition(ori_mask_image, resolution)

                        # Generate image using control net inpainting
                        result_images = sd_inpaint(init_image, mask_image, prompt, negative_prompt)
                        #print(f'results images are {result_images}')


                        for image in enumerate(result_images):
                            #print(image)
                            #JSON
                            random_number = random.randint(1000, 9999) # Create a unique name to avoid overwriting previous images
                            output_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))}_{category}_{random_number}_{os.path.splitext(image_path)[1]}")


                            image[1].save(output_file)

                            #result_image_cropped.save(cropped_file)
                            #mask_image.save(mask_file)



# Define the Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter prompt..."),
        gr.inputs.Textbox(lines=2,
                          default="hat, big breast, earring, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"),
        gr.inputs.Textbox(lines=2, default="/home/heran/Documents/easy_wild/women"),
        gr.inputs.Textbox(lines=2, placeholder="Enter output path..."),
        gr.inputs.Checkbox(label="Change Hair", default=True),
    ],
    outputs="text"
)

iface.launch()

# XpucT/Deliberate
# Lykon/DreamShaper
# hands, hat, big breast, long neck, BadDream, UnrealisticDream, ear, earring
