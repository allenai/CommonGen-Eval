import json 

with open("eval_outputs/aggregated_eval_results.json", "r") as f:
    data = json.load(f)



def visualize_case(item):
    print(f"- Concepts: {item['concept_set']}")
    print(f"- Human Reference: {item['human_ref']}")
    for model_name, o in item["model_outputs"].items():
        print(f"- {model_name}: {o['output']}")
        print(f"\t\t ----> vs Human = {o['result']}")
    
visualize_case(data["f8efb2bbd593ee6e943cfa79aa347e2e#1"])
