import PIL
import torch

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, EulerAncestralDiscreteScheduler

init_image = PIL.Image.open('image2.png').convert("RGB")
mask_image = PIL.Image.open('mask2.png').convert("RGB")
prompt = "(high detailed hair:1.2), RAW photo, a close up portrait photo, 8k uhd, dslr, soft \
lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, \
drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low \
quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, \
mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, \
blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
extra arms, extra legs, fused fingers, too many fingers, long neck"
init_image = init_image.resize((1024, 1024))
mask_image = PIL.ImageOps.invert(mask_image.resize((1024, 1024)))

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "XpucT/Deliberate", controlnet=controlnet, torch_dtype=torch.float16
).to('cuda')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt=prompt, negative_prompt=negative_prompt, control_image=init_image, image=init_image, \
                mask_image=mask_image, num_inference_steps=20, guidance_scale=7, generator=torch.Generator(device="cuda").manual_seed(-1)).images[0]
image.show()
image.save('result.png')