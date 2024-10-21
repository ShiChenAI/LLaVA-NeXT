# -*- coding: utf-8 -*-
"""
@File        :   process_cauldronja.py
@Author      :   Shi Chen
@Time        :   2024/10/21
@Description :   Convert Cauldron-JA in *.parquet format to *.jsonl format.
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import jsonlines
from PIL import Image
import io
from tqdm import tqdm
import base64

def get_args():
    parser = argparse.ArgumentParser(description='Convert dataset format')
    parser.add_argument('--data-dir', default="./datasets/temp/Cauldron-JA", type=str, help='Directory of the original dataset')
    parser.add_argument('--extra-image-dir', default="/storage_nvme/datasets/geniac2.0-multi-modal-datasets/datasets/pair_datasets/Cauldron-JA/extra_images/", type=str, help='Directory of extra images')
    parser.add_argument('--output-image-dir', default="./datasets/temp/test/images", type=str, help='Output directory for images')
    parser.add_argument('--output-jsonl-file', default="./datasets/temp/test/a.jsonl", type=str,help='Output JSONL file')
    parser.add_argument('--image-error-file', default="./datasets/temp/test/i.jsonl", type=str, help='File to log image extraction errors')
    parser.add_argument('--text-error-file', default="./datasets/temp/test/t.jsonl", type=str, help='File to log text extraction errors')
    return parser.parse_args()

def make_serializable(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == 'bytes':
                # Encode the 'bytes' field using base64
                if isinstance(v, (bytes, bytearray)):
                    encoded_bytes = base64.b64encode(v).decode('utf-8')
                    new_obj[k] = encoded_bytes
                elif isinstance(v, list):
                    # Convert list of integers to bytes before encoding
                    byte_data = bytes(v)
                    encoded_bytes = base64.b64encode(byte_data).decode('utf-8')
                    new_obj[k] = encoded_bytes
                elif isinstance(v, np.ndarray):
                    # Convert numpy array to bytes before encoding
                    byte_data = v.tobytes()
                    encoded_bytes = base64.b64encode(byte_data).decode('utf-8')
                    new_obj[k] = encoded_bytes
                else:
                    # If it's another type, attempt to serialize
                    new_obj[k] = make_serializable(v)
            else:
                new_obj[k] = make_serializable(v)
        return new_obj
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, bytes):
        # Encode bytes using base64
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    else:
        return obj

def extract_target_file(path):
    # Normalize the path to handle different OS path separators
    path = os.path.normpath(path)
    # Split the path into components
    components = path.split(os.sep)
    try:
        # Find the index of 'extracted'
        idx = components.index('extracted')
        # The target path is after the hash directory (which is at idx + 1)
        target_components = components[idx + 2:]
        # Join the components back into a path
        target_path = os.path.join(*target_components)
        return target_path
    except ValueError:
        # 'extracted' not found in the path
        return None
    
def run(args):
    data_dir = args.data_dir
    extra_image_dir = args.extra_image_dir
    output_image_dir = args.output_image_dir
    output_jsonl_file = args.output_jsonl_file
    image_error_file = args.image_error_file
    text_error_file = args.text_error_file

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    output_jsonl_dir = os.path.dirname(output_jsonl_file)
    if not os.path.exists(output_jsonl_dir):
        os.makedirs(output_jsonl_dir)

    image_id_counter = 0

    parquet_files = glob.glob(os.path.join(data_dir, '**/*.parquet'), recursive=True)

    # Calculate total number of samples for progress bar
    total_samples = 0
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=[])
        total_samples += table.num_rows

    # Open error log files
    image_error_writer = jsonlines.open(image_error_file, mode='w')
    text_error_writer = jsonlines.open(text_error_file, mode='w')

    # Write to JSONL file with tqdm progress monitoring
    with jsonlines.open(output_jsonl_file, mode='w') as writer, tqdm(total=total_samples, desc='Processing Samples') as pbar:
        for parquet_file in parquet_files:
            # Get the subfolder name relative to data_dir
            rel_path = os.path.relpath(os.path.dirname(parquet_file), data_dir)
            subfolder_name = rel_path.replace('\\', '/')

            # Read the .parquet file
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            # Iterate over each sample
            for _, row in df.iterrows():
                image_saved = False
                image_id_counter += 1
                image_id_str = f'{image_id_counter:08d}'
                # Save the image and get the result
                try:
                    images_field = row['images']
                    # Check if images_field is a non-empty list and contains 'bytes'
                    if images_field and isinstance(images_field, np.ndarray) and 'bytes' in images_field[0] and 'path' in images_field[0]:
                        if images_field.shape[0] > 1:
                            print(f"id: {image_id_str}, shape: {images_field.shape}")
                        
                        image_info = images_field[0]
                        image = None
                        # Try to get image bytes from 'bytes' key
                        if 'bytes' in image_info and image_info['bytes']:
                            image_bytes_data = image_info['bytes']
                            # Handle different types of image_bytes_data
                            if isinstance(image_bytes_data, list):
                                image_bytes = bytes(image_bytes_data)
                            elif isinstance(image_bytes_data, bytes):
                                image_bytes = image_bytes_data
                            elif isinstance(image_bytes_data, np.ndarray):
                                image_bytes = image_bytes_data.tobytes()
                            else:
                                raise ValueError(f"Unsupported image bytes data type for ID {image_id_str}.")
                            # Convert bytes to an image
                            image = Image.open(io.BytesIO(image_bytes))
                        elif 'path' in image_info and image_info['path']:
                            # Read image from the provided path
                            image_path = os.path.join(extra_image_dir, extract_target_file(image_info['path']))
                            # Try to open the image from the given path
                            if os.path.exists(image_path):
                                image = Image.open(image_path)
                            else:
                                raise FileNotFoundError(f"Image file not found at path {image_path} for ID {image_id_str}")
                        else:
                            raise ValueError(f"No valid image data for ID {image_id_str}.")

                        # Convert image to RGB if not already
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        # Check and create the image subdirectory if it does not exist
                        image_subdir = os.path.join(output_image_dir, subfolder_name)
                        if not os.path.exists(image_subdir):
                            os.makedirs(image_subdir)
                        # Save the image as JPEG
                        image_filename = os.path.join(image_subdir, f'{image_id_str}.jpg')
                        image.save(image_filename, 'JPEG')
                        image_saved = True
                    else:
                        raise ValueError(f"No valid image data for ID {image_id_str}.")
                except Exception as e:
                    print(f"Error processing image ID {image_id_str}: {e}")
                    # Log the full sample to images_error.jsonl
                    error_sample = {'id': image_id_str, 'images': row['images'], 'texts': row['texts']}
                    serializable_error_sample = make_serializable(error_sample)
                    image_error_writer.write(serializable_error_sample)
                    image_saved = False

                # Create JSON record and write to JSONL file
                try:
                    texts_field = row['texts']
                    # Build the "conversations" field
                    conversations = []
                    for text_entry in texts_field:
                        # Process user message
                        human_entry = {
                            'from': 'human',
                            'value': text_entry.get('user', ''),
                            'jp': text_entry.get('jp_user', '')
                        }
                        # Process assistant reply
                        assistant_entry = {
                            'from': 'gpt',
                            'value': text_entry.get('assistant', ''),
                            'jp': text_entry.get('jp_assistant', '')
                        }
                        conversations.extend([human_entry, assistant_entry])
                    # Create the JSON record
                    json_record = {
                        'id': image_id_str,
                        'image': f"{subfolder_name}/{image_id_str}.jpg" if image_saved else None,
                        'conversations': conversations
                    }
                    writer.write(json_record)
                except Exception as e:
                    print(f"Error processing text for ID {image_id_str}: {e}")
                    # Log the full sample to texts_error.jsonl
                    error_sample = {'id': image_id_str, 'images': row['images'], 'texts': row['texts']}
                    serializable_error_sample = make_serializable(error_sample)
                    text_error_writer.write(serializable_error_sample)
                pbar.update(1)  # Update the progress bar

    # Close error log files
    image_error_writer.close()
    text_error_writer.close()

    print("Processing completed.")

if __name__ == '__main__':
    args = get_args()
    run(args)
