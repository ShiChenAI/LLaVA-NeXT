# -*- coding: utf-8 -*-
"""
@File        :   process_datasets_offline.py
@Author      :   Shi Chen
@Time        :   2024/10/18
@Description :   Process all images in datasets offline.
"""

import ast
import re
import argparse
import json
import jsonlines
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from pathlib import Path
from dataclasses import dataclass, field
import torch
import transformers
from transformers import AutoConfig
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, resize_and_center_crop, extract_patches, process_image
from llava.model.builder import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llava_qwen", help="Name of the model.")
    parser.add_argument("--cfg-path", type=str, default="./checkpoints/qwen_05B/qwen_config.json", help="Path to the configuration file (*.json).")
    parser.add_argument("--ann-path", type=str, default="/storage_nvme/datasets/geniac2.0-multi-modal-datasets/datasets/pair_datasets/stage-2/LLaVA-v1.5-Instruct-620K-JA/llava_v1_5_instruct_620k_ja_v2.jsonl", help="Path to the annotation file.")
    parser.add_argument("--image-dir", type=str, default="/storage_nvme/datasets/geniac2.0-multi-modal-datasets/datasets/pair_datasets/stage-2/LLaVA-v1.5-Instruct-620K-JA/", help="Directory to the images.")
    parser.add_argument("--save-dir", type=str, default="./datasets/temp/620K/images", help="Directory to the processed images.")

    return parser.parse_args()

def run(args):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(args.cfg_path)
    #tokenizer, model, image_processor, max_length = load_pretrained_model(model_args.model_name_or_path, None, args.model_name, attn_implementation="sdpa")  # Add any other thing you want to pass in llava_model_args
    model = get_model(model_args, training_args, {})
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if "x" in data_args.image_grid_pinpoints:
        try:
            patch_size = data_args.image_processor.size[0]
        except Exception as e:
            patch_size = data_args.image_processor.size["shortest_edge"]

        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    else:
        data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.add_faster_video = model_args.add_faster_video
    model.config.faster_token_stride = model_args.faster_token_stride
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.ann_path) as f:
        for line in jsonlines.Reader(f):
            image_field = line["image"]
            image_names = [image_field] if isinstance(image_field, str) else image_field
            for image_name in image_names:
                image_path = os.path.join(args.image_dir, image_name)
                save_path = Path(args.save_dir) / Path(image_name).with_suffix(".pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as exn:
                    print(f"Failed to open image {image_path}. Exception:", exn)
                    raise exn
                image_tensor = process_image(image, data_args.image_processor, model.config)
                try:
                    torch.save(image_tensor, str(save_path))
                except Exception as e:
                    a = 0
                print(f"Saved to {str(save_path)}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})


def get_model(
    model_args, 
    training_args, 
    bnb_model_from_pretrained_args
):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model
"""
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(args.cfg_path)
    model = get_model(model_args, training_args, {})
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()

    image_processor = vision_tower.image_processor
    if "x" in image_grid_pinpoints:
        try:
            patch_size = image_processor.size[0]
        except Exception as e:
            patch_size = image_processor.size["shortest_edge"]

        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    else:
        image_grid_pinpoints = ast.literal_eval(image_grid_pinpoints)

    with open(args.ann_path) as f:
        for line in jsonlines.Reader(f):
            image_field = line["image"]
            image_names = [image_field] if isinstance(image_field, str) else image_field
            for image_name in image_names:
                image_path = os.path.join(args.image_dir, image_name)
                save_path = str(Path(args.save_dir) / Path(img_name).with_suffix(".pt").name)
                processed_image = 
            
"""


if __name__ == "__main__":
    args = parse_args()
    run(args)
