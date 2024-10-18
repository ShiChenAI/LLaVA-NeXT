# -*- coding: utf-8 -*-
"""
@File        :   process_datasets_offline.py
@Author      :   Shi Chen
@Time        :   2024/10/18
@Description :   Process all images in datasets offline.
"""

import argparse
from PIL import Image, ImageFile
from llava.mm_utils import process_highres_image, process_anyres_image, resize_and_center_crop, extract_patches

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", type=str, help="Path to the annotation file.")
    parser.add_argument("--img-dir", type=str, help="Directory to the images.")
    parser.add_argument("--save-dir", type=str, help="Directory to the processed images.")

    return parser.parse_args()

def process_highres_image_crop_split(
    image, 
    image_crop_resolution,
    image_split_resolution,
    processor
):
    image_crop = resize_and_center_crop(image, image_crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=image_split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]

    return torch.stack(image_patches, dim=0)

def process_image(
    image_file: str,
    image_folder: str,
    processor,
    image_aspect_ratio: str,
    image_grid_pinpoints: str,
    overwrite_image_aspect_ratio=None
):
    # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
    try:
        image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
    except Exception as exn:
        print(f"Failed to open image {image_file}. Exception:", exn)
        raise exn

    image_size = image.size
    if overwrite_image_aspect_ratio is not None:
        image_aspect_ratio = overwrite_image_aspect_ratio
    if image_aspect_ratio == "highres":
        image = process_highres_image(image, image_processor, image_grid_pinpoints)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        image = process_anyres_image(image, image_processor, image_grid_pinpoints)
    elif image_aspect_ratio == "crop_split":
        image = process_highres_image_crop_split(image, self.data_args)
    elif image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    
    return image, image_size, "image"

def run(args):
    return None

if __name__ == "__main__":
    args = parse_args()
    run(args)
