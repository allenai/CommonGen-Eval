# CommonGen-Eval
Evaluating LLMs with the CommonGen Task


## Leaderboard 

| model                      |length |   cover |    pos | win   |
|----------------------------|-------|---------|--------|-------|
| human                      | 12.84 |  100.00 | 100.00 | -     |
| gpt-4-1106-preview         | 14.90 |   96.78 |  90.22 | 50.78 |
| gpt-4-0613                 | 14.13 |   97.67 |  90.44 | 50.11 |
| gpt-3.5-turbo              | 12.76 |   91.67 |  82.00 | 49.22 |
| Yi-34b-chat                | 13.45 |   79.56 |  72.89 | 39.44 |
| vicuna-13b-v1.5            | 15.02 |   85.89 |  77.56 | 27.11 |
| tulu-2-dpo-70b             | 17.89 |   89.44 |  80.67 | 23.00 |
| Yi-6b-chat                 | 13.32 |   71.22 |  62.11 | 21.89 |
| Mixtral-8x7B-Instruct-v0.1 | 20.15 |   84.44 |  72.11 | 17.89 |
| zephyr-7b-beta             | 15.76 |   81.33 |  69.56 | 16.89 |
| Llama-2-7b-chat-hf         | 16.06 |   88.44 |  73.22 | 15.22 |

- lengh: the number of words on average in the generated sentences
- cover: the percentage of examples where all given concepts are covered by model outputs 
- pos: the percentage of examples where the part-of-speech (PoS) of each given concept is correct in model outputs
- win: the percentage of examples where GPT-4-turbo prefers the model outputs over the human-written references

## Run model inference with CommonGen-lite 

- Dataset: [CommonGen-lite](https://huggingface.co/datasets/allenai/commongen_lite) 
- Scripts: see `scripts/{model_name}.sh`

For example, 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python vllm_infer.py \
    --data_name "commongen" \
    --model_name 01-ai/Yi-34b-chat --tensor_parallel_size 4  --dtype bfloat16 \
    --output_folder "model_outputs/" \
    --top_p 1 --temperature 0 --batch_size 8 --max_tokens 128
```

## Run GPT-4 based evaluation 

- Scripts: see `scripts/all_gpt_eval.sh`.

```bash
models=("zephyr-7b-beta" "tulu-2-dpo-70b" "vicuna-13b-v1.5" "Llama-2-7b-chat-hf" "Mixtral-8x7B-Instruct-v0.1" "Yi-34b-chat" "Yi-6b-chat" "gpt-3.5-turbo" "gpt-4-0613" "gpt-4-1106-preview")
for model in "${models[@]}"
do 
    python evaluate.py --mode "compare" \
        --model_output_file "model_outputs/${model}.json" \
        --eval_output_file "eval_outputs/${model}.eval_result.gpt-4-1106-preview.json" \
        --model gpt-4-1106-preview &
done
```