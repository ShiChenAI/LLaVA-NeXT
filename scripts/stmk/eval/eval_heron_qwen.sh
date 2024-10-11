CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "datasets/Japanese-Heron-Bench/images" \
    --questions-file-path "datasets/Japanese-Heron-Bench/questions_ja.jsonl" \
    --answers-file-path "datasets/heron_results/heron_results_final_stage_qwen.jsonl" \
    --model-path "checkpoints/qwen_05B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2.5-0.5B-final_stage_llava_instruct_150k_ja" \
    --model-name "llava_qwen" \
    --conv-template "qwen_2" \
    --temperature 0.6