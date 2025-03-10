import json

with open('dpo_dataset_slice4.json', 'r') as file:
    data = json.load(file)

for item in data:
    if len(item['image']) > 1:
        item['image'] = [item['image'][0]]
    
    if len(item['conversations']) >= 2:
        item['bbox'] = item['conversations'][1]['value']
        item['conversations'] = [item['conversations'][0], item['conversations'][-1]]

with open('dpo_dataset_slice4.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Save to viscot_sample_dpo_convert_bbox.json")