import json

with open('viscot_sample_20k.json', 'r') as f:
    data = json.load(f)

for item in data:
    item['image'] = [item['image'][0]]
    item['conversations'] = item['conversations'][:2]

with open('viscot_sample_20k_round1.json', 'w') as f:
    json.dump(data, f, indent=4)