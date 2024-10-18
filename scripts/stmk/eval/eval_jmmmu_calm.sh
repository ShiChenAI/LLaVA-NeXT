export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN="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"

TASK="jmmmu"
CKPT_PATH="cyberagent/llava-calm2-siglip"
#CKPT_PATH_CLEAN="${CKPT_PATH//\//_}"
CONV_TEMPLATE="llava_llama_2"
MODEL_NAME="llava_llama"
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

#mmbench_en_dev,mathvista_testmini,llava_in_the_wild,mmvet
accelerate launch --num_processes 1 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
