"""
pip install spacy
python -m spacy download en_core_web_sm
"""

from eval_utils import analyze_words
import json 
from datasets import load_dataset
from tabulate import tabulate

files = {
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


# print(len(all_ref_ids_to_test))
human_row = {}
# print("human-winrate: -")
# print("human-pos-ratio: ", sum(human_pos_ratios)/len(human_pos_ratios))
# print("human-cover-ratio: ", sum(human_cover_ratios)/len(human_cover_ratios))
# print("human-avg-len: ", sum(human_lens)/len(human_lens))
# print("-"*20)
human_row["model"] = "human"
human_row["win"] = "-"
human_row["win_tie"] = "100"
# human_row["tie"] = "-"
human_row["pos"] = f"{sum(human_pos_ratios)/len(human_pos_ratios)*100:.2f}"
human_row["cover"] = f"{sum(human_cover_ratios)/len(human_cover_ratios)*100:.2f}"
human_row["len"] = f"{sum(human_lens)/len(human_lens):.2f}"
human_row["overall"] = "100"
table.append(human_row)
 
    

model_data = {}
llama_correct = set()
zephyr_correct = set()

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
        # if f"{item['id']}#{item['ref_index']}" in all_ref_ids_to_test:
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
    # print("win_count", win_count)
    # print("human_win_count", human_win_count)
    # print("tie_count", tie_count)
    assert win_count + human_win_count + tie_count == len(all_ref_ids_to_test)

    # print(f"{model}-winrate: {win_count/len(all_ref_ids_to_test)}")
    # print(f"{model}-loserate: {human_win_count/len(all_ref_ids_to_test)}")
    # print(f"{model}-pos-ratio: {sum(pos_ratios)/len(pos_ratios)}")
    # print(f"{model}-cover-ratio: {sum(cover_ratios)/len(cover_ratios)}")
    # print(f"{model}-avg-len: {sum(lens)/len(lens)}")
    # print("-"*20)
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
     

