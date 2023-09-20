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

position_third = None

resolution = 896

global w_face, h_face, x_face, y_face




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


def get_head_mask(image_path):
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

    #print(f'face ratio is {factor_1}')

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

    pipe = StableDiffusionInpaintPipeline.from_single_file("/home/heran/realistic-vision-inpaint/realisticVisionV51_v51VAE-inpainting.safetensors", torch_dtype=torch.float16, use_safetensors=True).to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.load_lora_weights("/home/heran/realistic-vision-inpaint", weight_name="koreanDollLikeness.safetensors")

    # Disables safety checks
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
         num_inference_steps=20, guidance_scale=7.5, height=896, width=672, num_images_per_prompt=3, cross_attention_kwargs={"scale": 0.5},
         generator=torch.Generator(device="cuda").manual_seed(seed)).images

    return images


def generate_images(image, gender, style):

    # Load JSON file containing style data
    with open('10prompt_9_15.json', 'r') as f:
        style_data = json.load(f)

    print("Loaded styles:", style_data.keys())  # Debug line
    
    try:
        selected_style = style_data[style.lower()]
    except KeyError as e:
        print(f"KeyError: {e}")  # Debug line
        return Image.new('RGB', (100, 100), "red")  # Placeholder for error
    prompt = selected_style[f'prompt_{gender[0].lower()}']
    negative_prompt = selected_style[f'neg_prompt_{gender[0].lower()}']

    # Convert the input NumPy array to a PIL Image
    init_image = Image.fromarray(image).convert("RGB")
    ori_init_image = resize_for_condition_image(init_image, resolution)

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        ori_init_image.save(temp.name)
        temp.close()  # Close the file so it can be opened by another process

    # Pass the file path to get_head_mask
    ori_mask_image = get_head_mask(temp.name)

    image_grid = []

    ori_mask_image = get_head_mask(temp.name)
          

    if prompt is not None:

        set_random_third_position()

        init_image = resize_for_init_image(ori_init_image)
        mask_image = resize_composition(ori_mask_image, resolution)

        # Generate image using inpainting
        result_images = sd_inpaint(init_image, mask_image, prompt, negative_prompt)

        for image in enumerate(result_images):
            image_grid.append(image[1])

    os.remove(temp.name) # Clean up the temporary file

    # Concatenate the result images into a single image
    output_image = concatenate_images(image_grid)

    return output_image

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Image(source="upload"),
        gr.Radio(choices=["Man", "Woman"], label="Choose Gender", default="Woman"),
        gr.Dropdown(choices=["Winter", "Autumn", "Summer", "Gongzhu", "Songlin", "Xianxia", "Gucheng", "Guilin", "Zhengqi_pengke", "Mountain"], label="Choose Style"),
    ],
    outputs=gr.Image(type='pil', label="Output Image"),
)

iface.launch()
