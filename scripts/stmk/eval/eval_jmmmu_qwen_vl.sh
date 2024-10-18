export CUDA_VISIBLE_DEVICES=0,1

TASK="jmmmu"
CKPT_PATH="Qwen/Qwen2-VL-7B-Instruct"
#CKPT_PATH_CLEAN="${CKPT_PATH//\//_}"
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

#mmbench_en_dev,mathvista_testmini,llava_in_the_wild,mmvet
#accelerate launch --num_processes 1 -m lmms_eval \
#python -m lmms_eval \
accelerate launch --num_processes 2 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=$CKPT_PATH,use_flash_attention_2=False,device_map=auto \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
