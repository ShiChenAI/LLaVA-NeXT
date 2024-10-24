export OMP_NUM_THREADS=8
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=ens18
#export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1

LLM_VERSION="Qwen/Qwen2-0.5B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_2"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-final_stage_cauldron" 
echo "RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node=2 --nnodes=1 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./scripts/stmk/train/final_stage.yaml \
    --image_folder /storage_nvme/datasets/geniac2.0-multi-modal-datasets/datasets/pair_datasets/Cauldron-JA-jsonl/images \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "./checkpoints/final_stage/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
