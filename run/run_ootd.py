import os
from pathlib import Path
import sys
import torch
from PIL import Image

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import time
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
ootd_model_hd = OOTDiffusionHD(0)

openpose_model_dc = OpenPose(1)
parsing_model_dc = Parsing(1)
ootd_model_dc = OOTDiffusionDC(1)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']


example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')


def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category =2

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images

if __name__ == "__main__":
    vton_img = "/opt/apps/oot_diffusion/data/modelv0.png"
    garm_img = "/opt/apps/oot_diffusion/data/clothv0.jpg"
    n_samples = 1
    n_steps = 20
    image_scale = 1.0
    seed = 0

    images = process_dc(vton_img, garm_img, 'Upper-body', n_samples, n_steps, image_scale, seed)
    images[0].save('result_dc.png')
