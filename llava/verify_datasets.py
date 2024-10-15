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
    parser.add_argument("--ann-path", type=str, help="Path to the annotation file.")
    parser.add_argument("--img-dir", type=str, help="Directory to the images.")

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
            img_path = os.path.join(args.img_dir, line["image"])
            if not os.path.exists(img_path):
                print(f"Not found: {img_path}")
            else:
                if is_valid_image(img_path):
                    n_files += 1
                else:
                    print(f"Not valid: {img_path}")
        print(f"Number of verified files: {n_files}")

if __name__ == "__main__":
    args = parse_args()
    run(args)