import argparse
import glob
import os
import PIL
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import facer
from pathlib import Path
import tempfile
import gradio as gr

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, EulerAncestralDiscreteScheduler

from controlnet_aux import OpenposeDetector



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
    input_image = ImageOps.expand(input_image, padding)

    k = float(resolution) / max(H, W)
    H = W = int(round(H * k / 64.0)) * 64
    
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def get_head_mask(image_path, mask_blur = 50, include_hair=True):
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


    mask_img = mask.float().cpu().numpy()*255
    mask_img = mask_img.astype(np.uint8)
    mask_img = Image.fromarray(mask_img[0])
    mask_img = mask_img.convert("L")


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

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/DreamShaper", controlnet=controlnet, torch_dtype=torch.float16
    ).to('cuda')


    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, control_image=[control_openpose_image, control_inpaint_image], image=init_image, \
                strength = 1, mask_image=mask_image, \
                num_inference_steps=25, guidance_scale=7, height=init_image.size[0], width=init_image.size[1], 
                generator=torch.Generator(device="cuda").manual_seed(-1)).images[0]

    return image



def generate_images(prompts, negative_prompt, input_path, output_path, change_hair):

    # Split the prompts by line
    prompts = prompts.split('\n')

    # Get list of all image files in the input path
    image_files = glob.glob(os.path.join(input_path, "*"))
    
    # Process each image
    for image_path in image_files:
        # Generate result image
        init_image = Image.open(image_path).convert("RGB")
        init_image = resize_for_condition_image(init_image, 896)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            init_image.save(temp.name)
            temp.close()  # Close the file so it can be opened by another process

            mask_image = get_head_mask(temp.name, include_hair=change_hair)


            # For each prompt, generate an image
            for i, prompt in enumerate(prompts):
                # Generate image using control net inpainting
                result_image = sd_controlnet_inpaint(init_image, mask_image, prompt, negative_prompt)
                
                # Save the result image to the output path with a name including the prompt index
                output_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_prompt{i}{os.path.splitext(image_path)[1]}")
                result_image.save(output_file)

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter prompt..."),
        gr.inputs.Textbox(lines=2, default="nsfw, nudity, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"),
        gr.inputs.Textbox(lines=2, placeholder="Enter input path..."),
        gr.inputs.Textbox(lines=2, placeholder="Enter output path..."),
        gr.inputs.Checkbox(label="Change Hair"),
    ],
    outputs="text"
)

iface.launch()


#XpucT/Deliberate
#Lykon/DreamShaper
