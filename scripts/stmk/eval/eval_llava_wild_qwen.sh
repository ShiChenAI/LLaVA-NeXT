TEMPERATURE=0.5

CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "./datasets/llava-bench-in-the-wild-ja/images" \
    --questions-file-path "./datasets/llava-bench-in-the-wild-ja/questions_ja.jsonl" \
    --answers-file-path "./datasets/llava_wild_results/t${TEMPERATURE}/LLaVA-ov-Qwen2.5-0.5B-_final_stage.jsonl" \
    --model-path "checkpoints/qwen_05B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2.5-0.5B-final_stage_llava_instruct_150k_ja" \
    --model-name "llava_qwen" \
    --conv-template "qwen_2" \
    --temperature ${TEMPERATURE}
