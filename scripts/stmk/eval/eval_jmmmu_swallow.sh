export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes=1 -m lmms_eval \
        --model llava_onevision \
        --model_args "pretrained=./checkpoints/swallow_8B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-final_stage_620k,conv_template=llava_llama_3" \
        --tasks jmmmu \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples_suffix llava_ov_ft_swallow_jmmmu \
        --log_samples