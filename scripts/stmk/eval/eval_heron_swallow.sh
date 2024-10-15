CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "datasets/Japanese-Heron-Bench/images" \
    --questions-file-path "datasets/Japanese-Heron-Bench/questions_ja.jsonl" \
    --answers-file-path "datasets/heron_results/heron_results_final_stage_8B.jsonl" \
    --model-path "checkpoints/swallow_8B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-final_stage_620k" \
    --model-name "llava_llama" \
    --conv-template "swallow" \
    --temperature 0.6