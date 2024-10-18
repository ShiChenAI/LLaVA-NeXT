TEMPERATURE=0.5

CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "./datasets/llava-bench-in-the-wild-ja/images" \
    --questions-file-path "./datasets/llava-bench-in-the-wild-ja/questions_ja.jsonl" \
    --answers-file-path "./datasets/llava_wild_results/t${TEMPERATURE}/LLaVA-ov-qwen2-7B.jsonl" \
    --model-path "lmms-lab/llava-onevision-qwen2-7b-ov" \
    --model-name "llava_qwen" \
    --conv-template "qwen_1_5" \
    --temperature ${TEMPERATURE}
