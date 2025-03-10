import json

sample_datasets = {'flickr30k', 'gqa', 'openimages', 'textcap', 'v7w'}
delete_dataset = 'docvqa'

with open('viscot_363k.json', 'r') as f:
    data = json.load(f)

dataset_dict = {}
for item in data:
    dataset = item.get('dataset', '').lower()
    if dataset not in dataset_dict:
        dataset_dict[dataset] = []
    dataset_dict[dataset].append(item)

new_data = []

for dataset, items in dataset_dict.items():
    if dataset == delete_dataset:
        continue
    else:
        new_data.extend(items[10000:20000])

with open('dpo_dataset_slice4.json', 'w') as f:
    json.dump(new_data, f, indent=4)