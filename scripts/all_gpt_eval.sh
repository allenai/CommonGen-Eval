mkdir eval_outputs
models=("zephyr-7b-beta" "tulu-2-dpo-70b" "vicuna-13b-v1.5" "Llama-2-7b-chat-hf" "Mixtral-8x7B-Instruct-v0.1" "Yi-34b-chat" "Yi-6b-chat" "gpt-3.5-turbo" "gpt-4-0613" "gpt-4-1106-preview")
for model in "${models[@]}"
do 
    python evaluate.py --mode "compare" \
        --model_output_file "model_outputs/${model}.json" \
        --eval_output_file "eval_outputs/${model}.eval_result.gpt-4-1106-preview.json" \
        --model gpt-4-1106-preview &
done
