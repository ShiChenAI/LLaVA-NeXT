CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "datasets/Japanese-Heron-Bench/images" \
    --questions-file-path "datasets/Japanese-Heron-Bench/questions_ja.jsonl" \
    --answers-file-path "datasets/heron_results/t0/heron_results_llava-ov-qwen2-0.5B.jsonl" \
    --model-path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
    --model-name "llava_qwen" \
    --conv-template "qwen_2" \
    --temperature 0