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
 
<details>
<summary>Instruction Prompt Template  (2-shots prompting)</summary>

```markdown
# Instruction

Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

# Examples

## Example 1
- Concepts: "dog(noun), frisbee(noun), catch(verb), throw(verb)"
- Sentence: The dog catches the frisbee when the boy throws it into the air.

## Example 2
- Concepts: "apple(noun), place(verb), tree(noun), pick(verb)"
- Sentence: A girl picks some apples from a tree and places them into her basket.

# Your Task 

- Concepts: "{$concept_list}"
- Sentence: 
```

</details>


## Run GPT-4 based evaluation 

To make your model on the leaderboard, please create an issue or PR to submit the inference script. I'll run the following evaluation script and update the leaderboard. You will not need to run the evaluation script yourself (the script needs special access to a HF dataset.)

Scripts: see `scripts/all_gpt_eval.sh` and `evaluate.py` for more details.

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

<details>
<summary>Evaluation Prompt Template (Pairwise Comparison)</summary>
```markdown
# Data

Given several concepts (i.e., nouns or verbs), we ask models to write a short and simple sentence that contains *all* the required words. 
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

Concepts: "{$concept_list}"

Model A: "{$candidate_A}"

Model B: "{$candidate_B}"

# Your Task

Your task is to choose a better sentence from the two candidates. Decide which model's sentence is better in terms of the naturalness and commonness of the scenes they describe. 

## Rules: 
- A better sentence should describe a common scene in daily life, and all concepts should be used in a natural way.
- You should prefer sentences that use all given concepts with correct part-of-speech tags. 
- A simpler and shorter sentence is preferred if it describes the same scene as the other sentence.
- If you think both sentences are equally good or bad, please choose "tie".

Now, please output your choice ("A" or "B" or "tie").

Your choice: 
```
</details>


## Case studies 



## Links 

- Contact: [Bill Yuchen Lin](https://yuchenlin.xyz/)
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


