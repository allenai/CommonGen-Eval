import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from eval_utils import (
    retry_handler, 
    openai_chat_request, 
)
import numpy as np 
from data_utils import concept_list_str
from datasets import load_dataset
 
def get_args():
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--model_output_file", type=str, required=False) 
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)  
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-0314")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
     
    return args
        

def parse_result(result_str):
    if "neither" in result_str.lower():
        return "neither"
    elif "A" in result_str:
        return "A"
    elif "B" in result_str:
        return "B"
    elif "tie" in result_str:
        return "tie"
    else:
        return "Not Matched"
                    
def gpt_eval(results, args):
    # try to load the existing results from args.eval_output_file 
    if os.path.exists(args.eval_output_file):
        cnt = 0 
        with open(args.eval_output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
                cnt += 1
        print(f"loading {cnt} results from {args.eval_output_file}")
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine
        
    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        result = openai_chat_request(**kwargs)
        result = result[0]  
        return result
    
    # results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        if item["result"] != "N/A": 
            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            print(f"Skipping {ind} for {args.eval_output_file}")
            # skip the existing results
            continue
            
        openai_args["prompt"] = item["prompt"]
        try:
            result = api(ind, item, **openai_args)
            results[ind]["result"] = result
            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            r = results[ind]["parsed_result"]
            if r in ["A", "B"]:
                results[ind]["winner"] = item["assignment"][r]
            else:
                results[ind]["winner"] = r 
                
        except Exception as e:
            print(e)
            raise Exception("Failed!")
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.eval_output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.eval_output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    # if K > 0 and len(text.split(" ")) > K:
    #     text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    pass 
    return text
 
    
def placeholder_generation(args): 
    commongen_data = load_dataset("allenai/commongen_lite_eval", split="train")
    with open("eval_template.md") as f:
        eval_template = f.read() 
    results = []
    with open(args.model_output_file, 'r') as f:
        candidates = json.load(f) 
    id_to_references = {x["id"]: x["human_annotations"] for x in commongen_data}
    candidates = [c for c in candidates if c["id"] in id_to_references]
    references = [id_to_references[c["id"]] for c in candidates]
    assert len(candidates) == len(references)
            
    L = len(candidates)
    if args.end_idx < 0:
        args.end_idx = L

    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx:args.end_idx]
    references = references[args.start_idx:args.end_idx]
    
    results = []
    for item, human_annoations in zip(candidates, references):
        instruction = item["instruction"] 
        for ref_id, ref in enumerate(human_annoations):
            o = item["output"][0]
            # random decide which is A and which is B 
            d = {}
            d["id"] = item["id"]
            d["ref_index"] = ref_id
            d["input"] = instruction
            d["concept_set"] = item["concept_set"]            
            d["human_ref"] = ref 
            d["model_output"] = item["output"]
            d["generator"] = item["generator"]  
            d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
            
            if random.random() < 0.5:
                A = o
                B = ref["ref"]
                d["assignment"] = {"A": d["generator"], "B": "human"}
            else:
                A = ref["ref"]
                B = o
                d["assignment"] = {"A": "human", "B": d["generator"]}
            cs_str = concept_list_str(d["concept_set"])
            prompt = eval_template.replace("{$concept_list}", cs_str).replace("{$candidate_A}", A).replace("{$candidate_B}", B)
            d["prompt"] = prompt
            d["result"] = "N/A" 
            results.append(d)
    return results 


def main():
    random.seed(42)
    args = get_args()
     
    if args.mode.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2) 
    elif args.mode.startswith("compare"):
        results = placeholder_generation(args)
        results = gpt_eval(results, args) 
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
     