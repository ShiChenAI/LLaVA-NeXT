CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-path "checkpoints/images/sakura.jpg" \
    --question "写真の概要を簡潔明瞭に説明してください。" \
    --model-path "checkpoints/swallow_8B/mid_stage/llavanext-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-mlp2x_gelu-mid_llava_pretrain_ja_chat" \
    --model-name "llava_llama" \
    --conv-template "swallow" \
    --temperature 0.6