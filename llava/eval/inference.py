import argparse
import copy
import torch
from typing import Dict, List, Optional, Tuple, Union
import jsonlines

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.generation.streamers import TextStreamer
import json
import os
import math
from tqdm import tqdm

from transformers import AutoConfig
from threading import Thread

import cv2
import base64

from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="checkpoints/images/cat.jpg", help="Path to the image file.")
    parser.add_argument("--image-dir", type=str, help="Directory to the image files.")
    parser.add_argument("--questions-file-path", type=str, help="Path to the questions file.")
    parser.add_argument("--answers-file-path", type=str, help="Path to the answers file.")
    parser.add_argument("--question", type=str, default="猫の隣には何がありますか？", help="Question to the single image.")
    parser.add_argument("--model-path", type=str, default="checkpoints/swallow_8B/mid_stage/llavanext-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-mlp2x_gelu-mid_llava_pretrain_ja_chat", help="Path to the model.")
    parser.add_argument("--model-name", type=str, default="llava_llama", help="Name of the model.")
    parser.add_argument("--conv-template", type=str, default="swallow", help="Conversation template.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)

    return parser.parse_args()

def generate_question_prompt(
    question: str,
    conv_template: str
):
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    question_prompt = conv.get_prompt()
    print(question_prompt)
    
    return question_prompt, stop_str

def generate_response(
    model,
    tokenizer,
    question_prompt: str,
    image_tensors,
    image_sizes: List[int],
    temperature: float,
    top_p: float,
    num_beams: int,
    stop_str: str,
    device: str
) -> str:
    input_ids = tokenizer_image_token(
        question_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

def run(args):
    print(args)
    assert args.image_path or args.image_dir

    # Initialize the model
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, args.model_name, device_map="auto", attn_implementation="sdpa")

    if args.image_path:
        assert args.question
        image = Image.open(args.image_path)
        image_sizes = [image.size]
        image_tensors = [_image.to(dtype=torch.float16, device=args.device) \
            for _image in process_images([image], image_processor, model.config)]

        # Generate question prompt
        question_prompt, stop_str = generate_question_prompt(args.question, args.conv_template)

        # Generate answer
        response = generate_response(model, tokenizer, question_prompt, image_tensors,
            image_sizes, args.temperature, args.top_p, args.num_beams, stop_str, args.device
        )
        print(response)
        
    if args.image_dir:
        assert args.question_file_path and args.answers_file_path
        with open(args.question_file_path) as f:
            ans_f = open(args.answers_file_path, "w")
            for line in tqdm(jsonlines.Reader(f)):
                image_path = os.path.join(args.image_dir, line["image"])
                question = line["text"]
                image = Image.open(image_path)
                image_sizes = [image.size]
                image_tensors = [_image.to(dtype=torch.float16, device=args.device) \
                    for _image in process_images([image], image_processor, model.config)]

                # Generate question prompt
                question_prompt, stop_str = generate_question_prompt(question, args.conv_template)
                
                # Generate answer
                response = generate_response(model, tokenizer, question_prompt, image_tensors,
                    image_sizes, args.temperature, args.top_p, args.num_beams, stop_str, args.device
                )

                # Write results
                ans_f.write(json.dumps({
                    "question_id": line["question_id"],
                    "image": line["image"],
                    "question": question,
                    "category": line["category"],
                    "image_category": line["image_category"],
                    "answer": response
                }))
                ans_f.flush()
            ans_f.close()

if __name__ == "__main__":
    args = parse_args()
    run(args)
