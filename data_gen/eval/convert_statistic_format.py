## change the format to statistic
import json

file_name = "result_dpo_model_80kcoherent_gpt4o_viscot_2_3w_create_false_cache_convert.json"
with open(file_name, 'r') as f:
    original_data = json.load(f)
    
def convert_to_3f(s):
    arr = json.loads(s)

    formatted_arr = ["{:.3f}".format(num) for num in arr]

    result = "[" + ", ".join(formatted_arr) + "]"
    return result

def transform_data(original_data):
    transformed = []
    for pair in original_data:
        if len(pair) != 2:
            continue
        chosen_item = pair[0]
        rejected_item = pair[1]
        
        new_entry = {
            "images": [chosen_item["image"]],
            "conversations": [
                {
                    "from": "human",
                    "value": chosen_item["prompt"]
                }
            ],
            "chosen_step": {
                "from": "gpt",
                "value": convert_to_3f(chosen_item["bbox"])
            },
            "rejected_step": {
                "from": "gpt",
                "value": convert_to_3f(rejected_item["bbox"])
            }
,
            "chosen": {
                "from": "gpt",
                "value": chosen_item["response"]
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_item["response"]
            }
        }
        transformed.append(new_entry)
    return transformed

transformed_data = transform_data(original_data)
with open(file_name, 'w') as f:
    json.dump(transformed_data, f, indent=2)