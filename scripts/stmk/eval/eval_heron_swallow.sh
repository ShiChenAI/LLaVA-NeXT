CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "datasets/Japanese-Heron-Bench/images" \
    --questions-file-path "datasets/Japanese-Heron-Bench/questions_ja.jsonl" \
    --answers-file-path "datasets/heron_results/heron_results_mid_stage_8B_plain.jsonl" \
    --model-path "checkpoints/swallow_8B/mid_stage/llavanext-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-mlp2x_gelu-mid_llava_pretrain_ja_plain" \
    --model-name "llava_llama" \
    --conv-template "swallow" \
    --temperature 0.6