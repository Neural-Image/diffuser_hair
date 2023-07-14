import gradio as gr
import argparse
import glob
import os
import PIL
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import torch
import facer
from pathlib import Path
import tempfile
from datetime import datetime
import random

from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionImg2ImgPipeline, ControlNetModel, DDIMScheduler, EulerAncestralDiscreteScheduler

from controlnet_aux import OpenposeDetector
#from clothing_inpaint import get_head_mask, sd_controlnet_inpaint

backgrounds = [", park in background", ", trees in background", ", mountain in background", ", sea in background, blue sky"]

def calculate_face_proportion(PIL_image):
    # Convert PIL Image to OpenCV format
    img = np.array(PIL_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load the Haar cascade xml file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Calculate total face area
    face_area = np.sum([w * h for (x, y, w, h) in faces])
    
    # Calculate the proportion of the image occupied by faces
    height, width, _ = img.shape
    total_area = height * width
    face_proportion = face_area / total_area
    
    return face_proportion

def dilate_mask(mask, kernel_size):
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return dilated_mask

def erode_mask(mask, kernel_size):
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask


def sd_img2img(init_image, stength):

    prompt = "((woman)), soft lighting, 4K, Masterpiece, original facial features, high quality face, high quality hair, realistic"
    negative_prompt = "disfigured, bad art, deformed, blurry,  morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions"

    #src_img = Image.fromarray(init_image.astype(np.uint8))
    src_img = Image.fromarray(np.array(init_image).astype(np.uint8))
    src_img.thumbnail((1024,1024))

    #"DGSpitzer/Cyberpunk-Anime-Diffusion" "XpucT/Deliberate"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/home/heran/DreamShaper/",safety_checker=None, torch_dtype=torch.float16).to('cuda')

    generator = torch.Generator(device='cuda').manual_seed(1024)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt,image=src_img, strength=stength, guidance_scale=7.5, generator=generator).images[0]

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
    input_image = ImageOps.expand(input_image, padding)

    k = float(resolution) / max(H, W)
    H = W = int(round(H * k / 64.0)) * 64
    
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def get_head_mask(image_path, kernel_size, mask_blur = 40, include_hair=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device)  # optional "farl/celebm/448"
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    vis_seg_probs = seg_probs.argmax(dim=1)
    # Include face, (hair) and face parts in the mask
    #1: Face, 2: Left Eyebrow, 3: Right Eyebrow, 4: Left Eye, 5: Right Eye, 6: Nose, 7: Upper Lip, 8: Inner Mouth, 9: Lower Lip, 10: Hair

    mask = ((vis_seg_probs == 1) | 
            (vis_seg_probs == 2) | (vis_seg_probs == 3) |
            (vis_seg_probs == 4) | (vis_seg_probs == 5) |
            (vis_seg_probs == 6) | (vis_seg_probs == 7) |
            (vis_seg_probs == 8) | (vis_seg_probs == 9))

    if not include_hair:
        mask = mask | (vis_seg_probs == 10)

    #mask = (vis_seg_probs == 0)

    mask_img = mask.float().cpu().numpy()*255
    mask_img = mask_img.astype(np.uint8)
    mask_img = Image.fromarray(mask_img[0])
    mask_img = mask_img.convert("L")


    # Convert the PIL Image back to a numpy array
    mask_img = np.array(mask_img)
    mask_img = erode_mask(mask_img, kernel_size=kernel_size)

    # After dilation, convert back to PIL Image for blurring
    mask_img = Image.fromarray(mask_img)


    mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))
    mask_img = PIL.ImageOps.invert(mask_img)
    #mask_img.show()
    return mask_img


def sd_controlnet_inpaint(init_image, mask_image, prompt, negative_prompt):

    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_openpose_image = openpose(init_image, hand_and_face=True)


    control_inpaint_image = make_inpaint_condition(init_image, mask_image)


    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16),
    ]

    '''
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/DreamShaper", controlnet=controlnet, torch_dtype=torch.float16
    ).to('cuda')
    '''

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "/home/heran/DreamShaper/", controlnet=controlnet, torch_dtype=torch.float16
    ).to('cuda')


    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    seed = random.randint(0, 2**32 - 1)

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, control_image=[control_openpose_image, control_inpaint_image], image=init_image, \
                strength = 1, mask_image=mask_image, \
                num_inference_steps=20, guidance_scale=7, height=init_image.size[0], width=init_image.size[1], 
                generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    return image


def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)
    new_img = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0,y_offset))
        y_offset += img.height
    return new_img



def inpaint(image, include_hair, kernel_size, prompts, negative_prompt):
    # Convert the input NumPy array to a PIL Image
    init_image = Image.fromarray(image).convert("RGB")
    init_image = resize_for_condition_image(init_image, 896)

    #face_proportion = calculate_face_proportion(image)
    #print(face_proportion)

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        init_image.save(temp.name)
        temp.close()  # Close the file so it can be opened by another process

    # Now pass the file path to get_head_mask
    mask_image = get_head_mask(temp.name, kernel_size, include_hair=include_hair)
    #mask_img = resize_for_condition_image(mask_image, 896)

    # Split the prompts and negative_prompts by line
    prompts = prompts.split('\n')

    result_images = []

    print(init_image.size)
    print(mask_image.size)

    for idx, prompt in enumerate(prompts):
        prompt += random.choice(backgrounds)
        result_image = sd_controlnet_inpaint(init_image, mask_image, prompt, negative_prompt)
        #result_image = sd_img2img(result_image, 0.1)


        result_images.append(result_image)

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Save each individual image in the 'output/images' folder
        result_image.save(f'output/images/image_{idx+1}_{timestamp}.png', 'PNG')

    #openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    #condition_image = openpose(init_image, hand_and_face=True)
    result_images.append(mask_image)  # Append the condition image to the end of the list

    # Clean up the temporary file
    os.remove(temp.name)

    # Concatenate the result images into a single image
    output_image = concatenate_images(result_images)

    # Save the concatenated image in the 'output/image_grids' folder
    output_image.save(f'output/image_grids/concatenated_image_{timestamp}.png', 'PNG')

    return output_image



iface = gr.Interface(
    fn=inpaint, 
    inputs=[
        gr.inputs.Image(source="upload"),
        gr.inputs.Checkbox(label="Change Hair", default=True),
        gr.inputs.Slider(minimum=0, maximum=50, default=5, label="Erosion"),  # Add a new Slider input for kernel size
        gr.inputs.Textbox(lines=10, label="Prompts"),
        gr.inputs.Textbox(lines=4, label="Negative Prompts", default="big breast, earring, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"),
    ], 
    outputs=gr.outputs.Image(type='pil', label="Output Image"),
)

iface.launch()
