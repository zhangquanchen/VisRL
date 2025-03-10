## filter the generated visrl data
import json
import json
import torch
from torchvision.ops import box_iou
from PIL import Image

def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    x1, y1, x2, y2 = box
    if w > h:
        pad = (w - h) // 2
        y1 += pad
        y2 += pad
    else:
        pad = (h - w) // 2
        x1 += pad
        x2 += pad
    return [x1, y1, x2, y2]

def convert_and_compute_iou(pred_box, gt_box, w, h):
    try:
        pred_abs = [
            pred_box[0] * w,
            pred_box[1] * h,
            pred_box[2] * w,
            pred_box[3] * h
        ]
        gt_abs = [
            gt_box[0] * w,
            gt_box[1] * h,
            gt_box[2] * w,
            gt_box[3] * h
        ]
    
        pred_sq = box_xyxy_expand2square(pred_abs, w=w, h=h)
        gt_sq = box_xyxy_expand2square(gt_abs, w=w, h=h)
        
        pred_tensor = torch.tensor([pred_sq], dtype=torch.float)
        gt_tensor = torch.tensor([gt_sq], dtype=torch.float)
        
        iou = box_iou(pred_tensor, gt_tensor).item()
        return iou
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0
    
def process_data(input_data):
    accuracy_max_thred = 0.8
    accuracy_min_thred = 0.2
    accuracy_bbox_max_thred = 0.8
    accuracy_bbox_min_thred = 0.2
    processed_data = []
    
    for sublist in input_data:
        if not sublist:
            continue
        filtered_sublist = [item for item in sublist if item['accuracy'] is not None and item['accuracy_bbox'] is not None]
        
        if not filtered_sublist:
            continue
        
        accuracies = [item['accuracy'] for item in filtered_sublist]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        
        if min_acc > accuracy_min_thred or max_acc < accuracy_max_thred or max_acc == min_acc:
            continue
        
        max_item = None
        min_item = None
        for item in filtered_sublist:
            if item['accuracy'] == max_acc and max_item is None and item['accuracy_bbox'] >= accuracy_bbox_max_thred:
                max_item = item
            if item['accuracy'] == min_acc and min_item is None and item['accuracy_bbox'] <= accuracy_bbox_min_thred:
                min_item = item
            if max_item and min_item:
                break
        
        if max_item and min_item:
            new_sublist = [max_item, min_item]
            processed_data.append(new_sublist)
    
    return processed_data

if __name__ == "__main__":
    with open('dpo_dataset_slice2_self_evolution_cache.json', 'r') as f:
        data = json.load(f)
    
    result = process_data(data)
    
    with open('dpo_dataset_slice2_self_evolution_cache_convert.json', 'w') as f:
        json.dump(result, f, indent=2)