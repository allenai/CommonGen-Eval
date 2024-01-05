import json 

with open("eval_outputs/aggregated_eval_results.json", "r") as f:
    data = json.load(f)



def visualize_case(item):
    print("""<details>
    <summary> Example 1 </summary>\n""")
    print(f"- **Concepts**: `{item['concept_set']}`")
    print(f"- **Human Reference**: `{item['human_ref']}`")
    for model_name, o in item["model_outputs"].items():
        cover_all = len(o["found_words"]) == len(item["concept_set"])
        pos_all = len(o["found_pos_words"]) == len(item["concept_set"])
        print(f"- **{model_name}**: `{o['output']}`")
        print(f"--> ```Versus={o['result']}; Cover={cover_all}; POS={pos_all}; Len={o['len']}```")
    print("\n</details>\n\n")    

visualize_case(data["f8efb2bbd593ee6e943cfa79aa347e2e#1"])

visualize_case(data["de5f30479745102be44f5f1f572730bc#0"])

visualize_case(data["81ec84a05e29c1b0066ebbc778ef9ae5#0"])

visualize_case(data["7b78772b4f86d8834751a0d852c0152d#0"])

