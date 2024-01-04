python vllm_infer.py \
    --data_name "commongen" \
    --engine openai --model_name openai/gpt-4-0613 \
    --output_folder "model_outputs/" \
    --top_p 1 --temperature 0 --max_tokens 128 