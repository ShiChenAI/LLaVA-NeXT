CUDA_VISIBLE_DEVICES=0 python llava/eval/inference.py \
    --image-path "datasets/images/validation_Economics_1.png" \
    --question "以下のTDの貸借対照表を考えてください。<image 1> 誰かがTD銀行に¥70000を預けたとします。このデータを考慮すると、資金流通量が増加する最小額はいくらですか？\nA. 0\nB. 70000\nC. 140000\nD. 341800\n\n与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。" \
    --model-path "checkpoints/swallow_8B/final_stage/llava-onevision-google_siglip-so400m-patch14-384-tokyotech-llm_Llama-3-Swallow-8B-Instruct-v0.1-final_stage_620k" \
    --model-name "llava_llama" \
    --conv-template "swallow" \
    --temperature 0
