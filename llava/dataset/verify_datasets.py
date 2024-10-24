# -*- coding: utf-8 -*-
"""
@File        :   verify_datasets.py
@Author      :   Shi Chen
@Time        :   2024/10/15
@Description :   Verify the availability of all images in the dataset.
"""

import jsonlines
import os
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", type=str, default="/storage_nvme/s-chen/Cauldron-JA-jsonl/Cauldron-JA_2.jsonl", help="Path to the annotation file.")
    parser.add_argument("--img-dir", type=str, default="/storage_nvme/s-chen/Cauldron-JA-jsonl/images", help="Directory to the images.")

    return parser.parse_args()

def is_valid_image(
    img_path: str
) -> bool:
    try:
        is_valid = True

        # Open in binary format
        file_obj = open(img_path, 'rb')  
        buf = file_obj.read()
        if not buf.startswith(b'\xff\xd8'):
            # Starts with "\xff\xd8" or not
            is_valid = False
        #elif buf[6:10] in (b'JFIF', b'Exif'):  # ASCII codes for "JFIF" and "Exif"
        #    if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
        #        # Ends with "\xff\xd9" or not
        #        is_valid = False
        else:
            try:
                Image.open(file_obj).verify()
            except Exception as e:
                is_valid = False
    except Exception as e:
        return False
    return is_valid

def run(args):
    with open(args.ann_path) as f:
        n_files = 0
        for line in jsonlines.Reader(f):
            id = line["id"]
            # Verify the availability of all images in the dataset.
            images_field = line["image"]
            image_names = images_field if isinstance(images_field, list) else [images_field]
            b_img = False
            for image_name in image_names:
                image_path = os.path.join(args.img_dir, image_name)
                if not os.path.exists(image_path):
                    b_img = False
                    print(f"ID: {id}, image not found: {image_path}")
                else:
                    if is_valid_image(image_path):
                        b_img = True
                    else:
                        b_img = False
                        print(f"ID: {id}, image Not valid: {image_path}")
            # Verify the conversations
            b_text = False
            conversations = line["conversations"]
            for idx, conversation in enumerate(conversations):
                target_key = "jp" if "jp" in conversation.keys() else "value"
                if conversation["from"] == "human":
                    if idx == 0:
                        b_text = (conversation[target_key] 
                                and len(conversation[target_key]) > 0 
                                and conversation[target_key].count("<image>") == len(image_names))
                    else:
                        b_text = (conversation[target_key] 
                                and len(conversation[target_key]) > 0 
                                and conversation[target_key].count("<image>") == 0)
                elif conversation["from"] == "gpt":
                    b_text = conversation[target_key] and len(conversation[target_key]) > 0
            
            if not b_text:
                print(f"ID: {id}, Conversations not vaild: {conversations}")

            if b_img and b_text:
                n_files += 1

        print(f"Number of verified files: {n_files}")

if __name__ == "__main__":
    args = parse_args()
    run(args)