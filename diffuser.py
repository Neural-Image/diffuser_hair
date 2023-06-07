import PIL
import numpy as np
import torch
import facer

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, EulerAncestralDiscreteScheduler

def resize_for_condition_image(input_image: PIL.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
    return img

def get_hair_mask(image_path):
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
    mask_img = PIL.Image.fromarray(mask_img[0])
    mask_img.show()
    return mask_img

image_path = 'image2.png'
init_image = PIL.Image.open(image_path)
mask_image = get_hair_mask(image_path) #PIL.Image.open('mask2.png').convert("RGB")/255.0

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
image = pipe(prompt=prompt, negative_prompt=negative_prompt, control_image=cond_image, image=cond_image, \
                strength = 1, mask_image=mask_image, num_inference_steps=20, guidance_scale=7, height=1024, width=1024, generator=torch.Generator(device="cuda").manual_seed(-1)).images[0]
image.show()
image.save('result_' + image_path)