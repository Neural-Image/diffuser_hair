import argparse
import glob
import math
import os
from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2
import facer
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, \
    EulerAncestralDiscreteScheduler

WIDTH = 512
HEIGHT = 512

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="Input image folder path")
    parser.add_argument("-o", "--output_path", type=str, help="Result image folder path")
    args = parser.parse_args()
    return args

def erode_mask(mask, kernel_size):
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask

def dilate_mask(mask, kernel_size):
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def get_specific_mask(vis_seg_probs, type_class1, type_class2):
    mask = vis_seg_probs.detach().clone()
    mask[mask == type_class1] = 255   
    mask[mask == type_class2] = 255
    mask[mask != 255] = 0   
    mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    return mask

def get_hair_mask(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_aligner = facer.face_aligner('farl/ibug300w/448', device=device) # optional: "farl/wflw/448", "farl/aflw19/448"
    with torch.inference_mode():
        faces_a = face_aligner(image, faces)
    #img = cv2.imread(image_path)[..., ::-1]
    #vis_img = img.copy()
    for pts in faces_a['alignment']:
        pts = pts.cpu().numpy()
        left_half_face = pts[2:8,:]
        right_half_face = pts[9:15,:]
        jaw_point_y = int(pts[8:9,:][0][1])
        print(jaw_point_y)
        middle_face_y = pts[[2,14],:]
        middle_face_y = int((middle_face_y[0][1] + middle_face_y[1][1])//2)
        print(middle_face_y)
        face_left_point = pts[0:1,:]
        face_right_point = pts[16:17,:]
        face_horizontal_size = face_right_point[0][0] - face_left_point[0][0]
        pts = pts[9:15,:]
        divider = int(face_horizontal_size // 30)
        left_half_face[:,0] = left_half_face[:,0] + divider
        right_half_face[:,0] = right_half_face[:,0]- divider
        face_pts_rows = left_half_face[:,1].astype(int).tolist() + right_half_face[:,1].astype(int).tolist()
        face_pts_cols = right_half_face[:,0].astype(int).tolist() + right_half_face[:,0].astype(int).tolist()
        #vis_img = facer.draw_landmarks(vis_img, None, pts)
    
    face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
    with torch.inference_mode():
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    #n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1)#/n_classes*255
    # vis_seg_probs_new = vis_seg_probs/n_classes*255
    # vis_img_new = vis_seg_probs_new.sum(0, keepdim=True)
    # facer.show_bhw(vis_img_new)
    # facer.show_bchw(facer.draw_bchw(image, faces))

    #Get eye brow mask
    eye_brow_mask = get_specific_mask(vis_seg_probs, 2, 3)
    #Get eye mask
    eye_mask = get_specific_mask(vis_seg_probs, 4, 5)
    highest_point_of_eye_brow = min(np.where(eye_brow_mask==255)[1])
    lowest_point_of_eye = max(np.where(eye_mask==255)[1])
    #Get lower face mask 
    lower_face_mask = get_specific_mask(vis_seg_probs, 1, 1)
    lower_face_mask[:,:lowest_point_of_eye,:] = 0
    #Get upper face mask
    upper_face_mask = get_specific_mask(vis_seg_probs, 1, 1)
    upper_face_mask[:,highest_point_of_eye_brow:,:] = 0 
    face_pixel = np.count_nonzero(lower_face_mask == 255)
    #Get hair mask
    hair_mask = get_specific_mask(vis_seg_probs, 10, 10)
    #Check wheather 12 face points r in hair region
    face_points = np.zeros_like(hair_mask)
    face_points[:,face_pts_rows,face_pts_cols] = 1
    face_points_in_hair = face_points * hair_mask
    face_points_in_hair = np.count_nonzero(face_points_in_hair == 255)
    if face_points_in_hair > 6:
        print(f"face points in hair is: {face_points_in_hair} > 6, skip")
        return None
    hair_count = np.count_nonzero(hair_mask == 255)
    lower_hair_count = np.count_nonzero(hair_mask[:,middle_face_y:jaw_point_y,:] == 255)
    lower_hair_pp = lower_hair_count / hair_count
    dilate_kernal_size = int(math.sqrt(face_pixel)) // divider
    if lower_hair_pp > 0.1:
        mask_dilate_size = int(math.sqrt(hair_count))//divider #dilate_kernal_size
    else:
        mask_dilate_size = int(math.sqrt(hair_count))//(divider//2) + dilate_kernal_size 
    mask_img = np.expand_dims(dilate_mask(np.squeeze(hair_mask[0]), mask_dilate_size), 0)
    eye_mask = np.expand_dims(dilate_mask(np.squeeze(eye_mask[0]), dilate_kernal_size//2), 0)
    lower_face_mask = np.expand_dims(erode_mask(np.squeeze(lower_face_mask[0]), int(2* dilate_kernal_size)), 0)
    #Mask out the eye area in the final mask
    mask_img[upper_face_mask==255] = 255
    mask_img[eye_mask == 255] = 0
    mask_img[eye_brow_mask == 255] = 0
    mask_img[lower_face_mask == 255] = 0
    mask_img = Image.fromarray(mask_img[0])
    mask_img = mask_img.convert("L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(dilate_kernal_size//2))
    return mask_img

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def sd_controlnet_inpaint(init_image, mask_image):
    prompt = "(round bald head, hairless, smoothy head skin, natural skin:1.2), RAW photo, a close up portrait photo, 8k uhd, dslr, soft \
    lighting, high quality, film grain, no hair on the scalp, Fujifilm XT3"

    negative_prompt = "(hair, hat, eyes, wrinkle, bright forehead skin, strong lighting, black contour, skin spots, black edges, skin black wrinkles:2), deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, \
    drawing, anime,  text, close up, cropped, out of frame, worst quality, low \
    quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, \
    mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, \
    blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
    disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
    extra arms, extra legs, fused fingers, too many fingers, long neck"
    control_image = make_inpaint_condition(init_image, mask_image)
    controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)   
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "XpucT/Deliberate", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, \
                    strength = 1, mask_image=mask_image, \
                    control_image=control_image, \
                    num_inference_steps=30, guidance_scale=7.5, height=HEIGHT, width=WIDTH, \
                    generator=torch.Generator(device="cuda").manual_seed(-1)).images[0]
    return image

def generate_final_result(image_path):
    init_image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
    mask_image = get_hair_mask(image_path)
    if mask_image == None:
        return None
    mask_image = mask_image.resize((WIDTH, HEIGHT))
    result_image = sd_controlnet_inpaint(init_image, mask_image)
    mask = np.array(mask_image.resize((WIDTH, HEIGHT)))/255.0
    mask = np.expand_dims(mask, axis=2)
    result = mask * np.array(result_image) + (1 - mask) * np.array(init_image.resize((WIDTH,HEIGHT)))
    result = Image.fromarray(result.astype(np.uint8))
    return result

if __name__=="__main__":
    args = parse_agrs()
    image_files  = glob.glob(args.image_path + '*.png')
    os.makedirs(args.output_path, exist_ok=True)
    for image_file in image_files:
        print(image_file)
        result =  generate_final_result(image_file)
        if result == None:
            continue
        file_name = os.path.basename(image_file)
        result_file_name = os.path.join(args.output_path, file_name)
        result.save(result_file_name)