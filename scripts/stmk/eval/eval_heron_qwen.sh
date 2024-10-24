CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "datasets/Japanese-Heron-Bench/images" \
    --questions-file-path "datasets/Japanese-Heron-Bench/questions_ja.jsonl" \
    --answers-file-path "datasets/heron_results/t0/heron_results_mid_stage_qwen.jsonl" \
    --model-path "checkpoints/qwen_05B/mid_stage/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2.5-0.5B-mlp2x_gelu-mid_llava_pretrain_ja_plain" \
    --model-name "llava_qwen" \
    --conv-template "qwen_2" \
    --temperature 0