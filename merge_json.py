import json
import sys 
import os
# find all json files under the `vllm_outputs/commongen/`, and only keep the ones starts with the string sys.argv[1]
folder = sys.argv[1]
prefix = sys.argv[2]
json_files = []
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.startswith(prefix) and file.endswith('.json'):
            # there must be a "-" between prefix and ".json" in the filename 
            if "-" in file.replace(prefix, "").replace(root, ""):
                json_files.append(os.path.join(root, file))
            
# merge all json files into one
data = []
for json_file in json_files:
    with open(json_file) as f:
        cur_data = json.load(f)
        # if "gpt" in prefix:
        #     # if model outputs does not ends with "." then we append it 
        #     for item in cur_data:
        #         for ind in range(len(item["output"])):
        #             if not item["output"][ind].endswith("."):
        #                 item["output"][ind] = item["output"][ind] + "."
        data.extend(cur_data)

# save the merged json file
with open(f"{folder}/{prefix}.json", 'w') as f:
    json.dump(data, f, indent=2)