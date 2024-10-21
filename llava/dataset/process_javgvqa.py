# -*- coding: utf-8 -*-
"""
@File        :   process_javgvqa.py
@Author      :   Shi Chen
@Time        :   2024/10/21
@Description :   Convert ja-vg-vqa dataset to *.jsonl format.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Process dataset and convert format.")
    parser.add_argument("--images-dir", type=str, help="Directory containing the images.")
    parser.add_argument("--annotations-file", type=str, help="Path to the annotations JSON file.")
    parser.add_argument("--subfolders", type=str, nargs='+', default=["VG_100K", "VG_100K_2"], help="List of subfolders in images directory.")
    parser.add_argument("--output-images-dir", type=str, help="Output directory for images.")
    parser.add_argument("--output-jsonl-file", type=str, help="Output JSONL file path.")
    args = parser.parse_args()
    return args

def run(args):
    images_dir = args.images_dir
    subfolders = args.subfolders
    output_images_dir = args.output_images_dir
    output_jsonl_file = args.output_jsonl_file
    annotations_file = args.annotations_file

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    output_jsonl_dir = os.path.dirname(output_jsonl_file)
    if not os.path.exists(output_jsonl_dir):
        os.makedirs(output_jsonl_dir)

    # Read the annotation JSON data
    with open(annotations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize sample ID counter
    sample_id_counter = 0

    # Open the output JSONL file
    with open(output_jsonl_file, "w", encoding="utf-8") as out_f:
        for entry in tqdm(data, desc="Processing entries"):
            # Get the image ID
            image_id = entry["id"]

            qas = entry.get("qas", [])

            # Skip if there are no Q&A pairs
            if not qas:
                print(f"No Q&A pairs: {image_id}")
                continue

            # Build the conversations list
            conversations = []
            for qa in qas:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                conversations.append({"from": "human", "value": question})
                conversations.append({"from": "gpt", "value": answer})

            # Increment sample ID counter
            sample_id_counter += 1
            output_id = str(sample_id_counter)

            # Find and copy the image
            found = False
            for subfolder in subfolders:
                src_image_path = os.path.join(images_dir, subfolder, f"{image_id}.jpg")
                if os.path.exists(src_image_path):
                    dst_image_path = os.path.join(output_images_dir, subfolder, f"{image_id}.jpg")
                    if not os.path.exists(os.path.dirname(dst_image_path)):
                        os.makedirs(os.path.dirname(dst_image_path))
                    shutil.copy(src_image_path, dst_image_path)
                    found = True
                    image_relative_path = os.path.join(subfolder, f"{image_id}.jpg")
                    break
            if not found:
                print(f"Image {image_id} not found in any subfolder.")
                continue  # Skip this entry if image not found

            # Build the output JSON object
            output_obj = {
                "id": output_id,
                "image": image_relative_path,
                "conversations": conversations
            }

            # Write the JSON object to the output file
            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = get_args()
    run(args)
