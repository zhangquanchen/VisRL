import json
from PIL import Image, ImageDraw
import os

def draw_bounding_boxes(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['green', 'purple']
    
    for group_idx, group in enumerate(data):
        if not group:
            continue
        
        base_image_path = group[0]['image']
        
        try:
            img = Image.open(base_image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {base_image_path}")
            continue
        
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        for item_idx, item in enumerate(group):
            try:
                bbox = json.loads(item['bbox'])
                if len(bbox) != 4:
                    raise ValueError("Invalid bbox format")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing bbox in group {group_idx}, item {item_idx}: {e}")
                continue
            
            x_min = bbox[0] * img_width
            y_min = bbox[1] * img_height
            x_max = bbox[2] * img_width
            y_max = bbox[3] * img_height
            
            color = colors[item_idx]
            print(color)
            print(item)
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=3
            )


        bbox = json.loads(item['gt_bbox'])
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        x_max = bbox[2] * img_width
        y_max = bbox[3] * img_height
        color = 'red'
        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline=color,
            width=3
        )
        
        filename = os.path.basename(base_image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_annotated{ext}")
        
        img.save(output_path)
        print(f"Successfully saved annotated image: {output_path}")

if __name__ == "__main__":
    draw_bounding_boxes(
        json_file='output.json',
        output_dir='dpo_bbox_annotated'
    )