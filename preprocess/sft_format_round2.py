import json
from PIL import Image
import os

def crop_image(input_path, output_path, bbox):
    try:
        with Image.open(input_path) as img:
            img_width, img_height = img.size
            left = max(0, min(bbox[0], img_width))
            top = max(0, min(bbox[1], img_height))
            right = max(left, min(bbox[2], img_width))
            bottom = max(top, min(bbox[3], img_height))
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return False

def process_data(data):
    for item in data:
        img_path, img_with_bbox = item['image']
        parts = img_with_bbox.split('###')
        if len(parts) != 2:
            print(f"Invalid image entry: {img_with_bbox}")
            continue
        img_name = parts[0]
        bbox_str = parts[1].strip('[]')
        try:
            bbox = list(map(int, bbox_str.split(',')))
        except:
            print(f"Invalid bounding box coordinates: {bbox_str}")
            continue
        if len(bbox) != 4:
            print(f"Invalid bounding box format: {bbox}")
            continue
        new_img_name = f"{os.path.splitext(img_name)[0]}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"
        if not crop_image(img_name, new_img_name, bbox):
            print(f"Failed to crop image {img_name}. Skipping this data entry.")
            continue
        item['image'][1] = new_img_name
        if len(item['conversations']) < 4:
            print("Conversations field does not have enough entries. Skipping this data entry.")
            continue
        human_msg = item['conversations'][0]['value']
        text_to_remove = "Please provide the bounding box coordinate of the region that can help you answer the question better."
        human_msg = human_msg.replace(text_to_remove, "Please first provide the bounding box coordinate of the region, then refer to the corresponding sub-image to answer the question better.")
        human_msg += "\n<image>"
        last_gpt = item['conversations'][-1]['value']
        item['conversations'] = [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": last_gpt}
        ]
    return data

with open('viscot_sample_20k.json', 'r') as f:
    data = json.load(f)

processed_data = process_data(data)

with open('viscot_sample_20k_round2.json', 'w') as f:
    json.dump(processed_data, f, indent=4)
    
print('Successfully!')