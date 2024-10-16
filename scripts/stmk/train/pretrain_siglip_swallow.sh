#export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eno1
#export NCCL_P2P_DISABLE=1
#export NCCL_DEBUG=INFO
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_LAUNCH_BLOCKING=1 
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10240

LLM_VERSION="tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_595k_jp_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

#ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node=8 --nnodes=1 --node_rank=0 \
torchrun --standalone --nproc_per_node=8 --nnodes=1 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./datasets/cc3m_pretrain_595k_ja.json \
    --image_folder ./datasets/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa
