TEMPERATURE=0.5

CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-dir "./datasets/llava-bench-in-the-wild-ja/images" \
    --questions-file-path "./datasets/llava-bench-in-the-wild-ja/questions_ja.jsonl" \
    --answers-file-path "./datasets/llava_wild_results/t${TEMPERATURE}/LLaVA-ov-Swallow-8B_final_stage.jsonl" \
    --model-path "checkpoints/swallow_8B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-final_stage_620k" \
    --model-name "llava_llama" \
    --conv-template "swallow" \
    --temperature ${TEMPERATURE}
