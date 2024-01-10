from eval_utils import analyze_words
import json 
from datasets import load_dataset
from tabulate import tabulate
from collections import defaultdict
import copy 
files = {
    "Pallas-0.5": "eval_outputs/Pallas-0.5.eval_result.gpt-4-1106-preview.json",
    "Mixtral-8x7B-Instruct-v0.1": "eval_outputs/Mixtral-8x7B-Instruct-v0.1.eval_result.gpt-4-1106-preview.json",
    "Yi-34b-chat": "eval_outputs/Yi-34b-chat.eval_result.gpt-4-1106-preview.json",
    "Yi-6b-chat": "eval_outputs/Yi-6b-chat.eval_result.gpt-4-1106-preview.json",
    "Llama-2-7b-chat-hf": "eval_outputs/Llama-2-7b-chat-hf.eval_result.gpt-4-1106-preview.json",
    "zephyr-7b-beta": "eval_outputs/zephyr-7b-beta.eval_result.gpt-4-1106-preview.json", 
    "tulu-2-dpo-70b": "eval_outputs/tulu-2-dpo-70b.eval_result.gpt-4-1106-preview.json",
    "vicuna-13b-v1.5": "eval_outputs/vicuna-13b-v1.5.eval_result.gpt-4-1106-preview.json",
    "gpt-3.5-turbo": "eval_outputs/gpt-3.5-turbo.eval_result.gpt-4-1106-preview.json",
    "gpt-4-0613": "eval_outputs/gpt-4-0613.eval_result.gpt-4-1106-preview.json",
    "gpt-4-1106-preview": "eval_outputs/gpt-4-1106-preview.eval_result.gpt-4-1106-preview.json",
    }

table = []

aggregated_eval_results = {}
human_pos_ratios = []
human_cover_ratios = []
human_lens = []
all_ref_ids_to_test = []
truth_data = load_dataset("allenai/commongen_lite_eval", split="train")
for item in truth_data:
    for ref in item["human_annotations"]:
        all_ref_ids_to_test.append(ref["id"]) 
        cs = item["concept_set"]
        found_words, found_pos_words = analyze_words(cs, ref["ref"])
        human_pos_ratios.append(len(found_pos_words)==len(cs))
        human_cover_ratios.append(len(found_words)==len(cs))
        human_lens.append(len(ref["ref"].split(" ")))
        aggregated_eval_results[ref["id"]] = {"concept_set": item["concept_set"], "human_ref": ref["ref"], "found_words": list(found_words), "found_pos_words": list(found_pos_words), "len": len(ref["ref"].split(" ")), "model_outputs": {}}
 

# print(len(all_ref_ids_to_test))
human_row = {}
# print("human-winrate: -")
# print("human-pos-ratio: ", sum(human_pos_ratios)/len(human_pos_ratios))
# print("human-cover-ratio: ", sum(human_cover_ratios)/len(human_cover_ratios))
# print("human-avg-len: ", sum(human_lens)/len(human_lens))
# print("-"*20)
human_row["model"] = "human (upper bound)"
human_row["win"] = "-"
human_row["win_tie"] = "100.00"
# human_row["tie"] = "-"
human_row["pos"] = f"{sum(human_pos_ratios)/len(human_pos_ratios)*100:.2f}"
human_row["cover"] = f"{sum(human_cover_ratios)/len(human_cover_ratios)*100:.2f}"
human_row["len"] = f"{sum(human_lens)/len(human_lens):.2f}"
human_row["overall"] = f"{(float(human_row['win_tie']) * float(human_row['pos']) * float(human_row['cover'])) / 10000:.2f}"
table.append(human_row)
human_row_2 = copy.deepcopy(human_row)
human_row_2["model"] = "human (lower bound)"
human_row_2["win_tie"] = "50.00"
human_row["overall"] = f"{(float(human_row['win_tie']) * float(human_row['pos']) * float(human_row['cover'])) / 10000:.2f}"
table.append(human_row)
 
    

model_data = {}

# default dict to be a 0 

for model, file in files.items():
    # load the model outputs
    with open(file, "r") as f:
        results = json.load(f)
    model_data[model] = results
    win_count = 0
    human_win_count = 0
    tie_count = 0 
    pos_ratios = []
    cover_ratios = []
    lens = []
    print(len(results))
    for item in results: 
        if item["human_ref"]["id"] in all_ref_ids_to_test:
            if item["winner"] == model:
                win_count += 1    
            elif item["winner"] == "human":
                human_win_count += 1
            else:
                tie_count += 1
            output = item["model_output"][0]
            cs = item["concept_set"]
            found_words, found_pos_words = analyze_words(cs, output)
            pos_ratios.append(len(found_pos_words)==len(cs))
            cover_ratios.append(len(found_words)==len(cs))
            lens.append(len(output.split(" "))) 
        
        # if item["human_ref"]["id"] not in aggregated_eval_results:
            
        r = "win" if item["winner"] == model else "lose" if item["winner"] == "human" else "tie"
        aggregated_eval_results[item["human_ref"]["id"]]["model_outputs"][model] = {"output": output, "result": r, "found_pos_words": list(found_pos_words), "found_words": list(found_words), "len": len(output.split(" "))} 
            
    assert win_count + human_win_count + tie_count == len(all_ref_ids_to_test) 
    row = {}
    row["model"] = model
    row["win"] = f"{(win_count)/len(all_ref_ids_to_test)*100:.2f}"
    row["win_tie"] = f"{(win_count+tie_count)/len(all_ref_ids_to_test)*100:.2f}"
    # row["tie"] = tie_count/len(all_ref_ids_to_test)
    row["pos"] = f"{sum(pos_ratios)/len(pos_ratios)*100:.2f}"
    row["cover"] = f"{sum(cover_ratios)/len(cover_ratios)*100:.2f}"
    row["len"] = f"{sum(lens)/len(lens):.2f}"
    overall = (float(row["win_tie"]) * float(row["pos"]) * float(row["cover"])) / 10000
    row["overall"] = f"{overall:.2f}"
    table.append(row)
# sort by win
# order the keys by ["len", "cover", "pos", "win"]
table = [{"model": item["model"], "len": item["len"], "cover": item["cover"], "pos": item["pos"], "win_tie":item["win_tie"], "overall": item["overall"]} for item in table]
table = table[0:1] +  sorted(table[1:], key=lambda x: float(x["overall"]), reverse=True)
print(tabulate(table, headers="keys", tablefmt="github", floatfmt=".2f"))
print(tabulate(table, headers="keys", tablefmt="html", floatfmt=".2f").replace(' style="text-align: right;"', ""))
     

with open("eval_outputs/aggregated_eval_results.json", "w") as f:
    json.dump(aggregated_eval_results, f, indent=2)