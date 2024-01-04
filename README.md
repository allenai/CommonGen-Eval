# CommonGen-Eval
(Re-)Evaluating LLMs with the [CommonGen](https://inklab.usc.edu/CommonGen/) Task. We use 400 examples and use GPT-4 to evaluate model outputs versus human references. 


## Leaderboard 

| model                      |   len |   cover |   pos |   win_tie |   overall |
|----------------------------|-------|---------|-------|-----------|-----------|
| human                      | 12.84 |   99.00 | 98.11 |    100.00 |     97.13 |
| gpt-4-0613                 | 14.13 |   97.44 | 91.78 |     50.44 |     45.11 |
| gpt-4-1106-preview         | 14.90 |   96.33 | 90.11 |     50.78 |     44.08 |
| gpt-3.5-turbo              | 12.76 |   92.11 | 83.00 |     49.78 |     38.06 |
| Yi-34b-chat                | 13.45 |   80.11 | 75.11 |     39.44 |     23.73 |
| vicuna-13b-v1.5            | 15.02 |   85.89 | 79.56 |     27.44 |     18.75 |
| tulu-2-dpo-70b             | 17.89 |   88.78 | 80.11 |     23.00 |     16.36 |
| Mixtral-8x7B-Instruct-v0.1 | 20.15 |   84.11 | 73.33 |     17.89 |     11.03 |
| Llama-2-7b-chat-hf         | 16.06 |   88.56 | 76.44 |     15.44 |     10.45 |
| zephyr-7b-beta             | 15.76 |   82.44 | 72.78 |     16.89 |     10.13 |
| Yi-6b-chat                 | 13.32 |   71.67 | 63.56 |     22.11 |     10.07 |

- **length**: the number of words on average in the generated sentences
- **cover**: the percentage of examples where all given concepts are covered by model outputs 
- **PoS**: the percentage of examples where the part-of-speech (PoS) of ALL given concepts are correct in model outputs
- **win_tie**: the percentage of examples where GPT-4-turbo prefers the model outputs over the human-written references (or thinks they are equally good)
- **overall**: `cover%` x `pos%` x `win_tie%` 

## Installation 

```bash 
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Run model inference

- Dataset: [CommonGen-lite](https://huggingface.co/datasets/allenai/commongen_lite) 
- Scripts: see `scripts/{model_name}.sh`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python vllm_infer.py \
    --data_name "commongen" \
    --model_name 01-ai/Yi-34b-chat --tensor_parallel_size 4  --dtype bfloat16 \
    --output_folder "model_outputs/" \
    --top_p 1 --temperature 0 --batch_size 8 --max_tokens 128
```

## Run GPT-4 based evaluation 

- Dataset: you will need to apply for the access to [CommonGen-lite-eval](https://huggingface.co/datasets/allenai/commongen_lite_eval) 
- Scripts: see `scripts/all_gpt_eval.sh`.

Example: 
```bash
models=("zephyr-7b-beta" "tulu-2-dpo-70b" "vicuna-13b-v1.5")
for model in "${models[@]}"
do 
    python evaluate.py --mode "compare" \
        --model_output_file "model_outputs/${model}.json" \
        --eval_output_file "eval_outputs/${model}.eval_result.gpt-4-1106-preview.json" \
        --model gpt-4-1106-preview &
done
```

## Contact 

- Person: [Bill Yuchen Lin](https://yuchenlin.xyz/)
- Project website: [https://inklab.usc.edu/CommonGen/](https://inklab.usc.edu/CommonGen/)
- HuggingFace Dataset: [CommonGen-lite](https://huggingface.co/datasets/allenai/commongen_lite) 

## Citation 

```bibtex
@inproceedings{lin-etal-2020-commongen,
    title = "{C}ommon{G}en: A Constrained Text Generation Challenge for Generative Commonsense Reasoning",
    author = "Lin, Bill Yuchen  and
      Zhou, Wangchunshu  and
      Shen, Ming  and
      Zhou, Pei  and
      Bhagavatula, Chandra  and
      Choi, Yejin  and
      Ren, Xiang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.165",
    pages = "1823--1840", 
}
```


