from datasets import load_dataset
from tqdm import tqdm
from fastchat_conversation import get_conv_template
import json 

def apply_template(chat_history, model_name):
    model_inputs = [] 
    for chats in tqdm(chat_history, desc="Applying template", disable=True):
        if "tulu" in model_name.lower():
            conv = get_conv_template("tulu")
        elif "zephyr" in model_name.lower():
            conv = get_conv_template("zephyr")
        elif "llama-2" in model_name.lower():
            conv = get_conv_template("llama-2")
        elif "mixtral" in model_name.lower() or "mistral" in model_name.lower():
            conv = get_conv_template("mistral")
        elif "yi" in model_name.lower() and "chat" in model_name.lower():
            conv = get_conv_template("Yi-34b-chat")
        elif "vicuna" in model_name.lower():
            conv = get_conv_template("vicuna_v1.1")
        elif "gpt-" in model_name.lower():
            model_inputs.append(chats[0])
            continue
        else:
            print("ERROR: model_name not supported")
        for chat_id, chat in enumerate(chats):
            conv.append_message(conv.roles[chat_id%2], chat)
        conv.append_message(conv.roles[1], None)
        model_inputs.append(conv.get_prompt())
    return model_inputs
 
def load_eval_data(args, data_name=None, model_name=None):
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name    
    chat_history = []
    id_strs = []
    metadata = {}
    if data_name  == "commongen":
        dataset = load_dataset("allenai/commongen_lite", split="train") 
        metadata = {"id": [], "concept_set": []}
    else:
        print("ERROR: data_name not supported")
     
    for ind, item in enumerate(dataset):
        if data_name in ["alpaca_eval", "just_eval", "commongen"]:
            in_text = item["instruction"]    
            id_strs.append(item.get("id", str(ind)))
            chat_history.append([in_text])
        for key in metadata: 
            metadata[key].append(item[key])
    print("start applying template")
    model_inputs = apply_template(chat_history, model_name)
    return id_strs, chat_history, model_inputs, metadata




def clear_output(output, model_name):
    # if "tulu" in model_name.lower() or "zephyr" in model_name.lower():
    #     output = output.replace("<|assistant|>\n", "")
    pass
    if "llama-2-7b" in model_name.lower():
        if "\n\n" in output:
            output = output[output.index("\n\n"):].strip()
    return output


def save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath):
    formatted_outputs = []
    if args.data_name == "commongen":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["instruction"] = chat_history[ind][0]
            if type(outputs[ind]) == list:
                output_item["output"] = [clear_output(o.rstrip(), args.model_name) for o in outputs[ind]]
            elif type(outputs[ind]) == str:
                output_item["output"] = clear_output(outputs[ind].rstrip(), args.model_name)
            output_item["generator"] = args.model_name
            for key in metadata:
                output_item[key] = metadata[key][ind]
            formatted_outputs.append(output_item) 
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)
        

def concept_list_str(concept_set):
    concept_strs = []
    for concept in concept_set:
        concept_strs.append(concept.replace("_N", "(noun)").replace("_V", "(verb)"))
    return ", ".join(concept_strs)