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

def main():
    with open('dpo_dataset_slice2_self_evolution_cache_convert.json', 'r') as f:
        data = json.load(f)
    print(len(data))
    
    stats = {
        'TP_TN': 0,
        'TP_FN': 0,
        'FP_FN': 0,
        'FP_TN': 0 
    }
    
    for group in data:
        if len(group) < 2:
            continue
        
        try:
            img_path = group[0]['image']
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        pos = group[0]
        try:
            pos_box = json.loads(pos['bbox'])
            pos_gt = json.loads(pos['gt_bbox'])
            pos_iou = convert_and_compute_iou(pos_box, pos_gt, w, h)
            pos_correct = pos_iou > 0.5
        except:
            pos_correct = False
        
        neg = group[1]
        try:
            neg_box = json.loads(neg['bbox'])
            neg_gt = json.loads(neg['gt_bbox'])
            neg_iou = convert_and_compute_iou(neg_box, neg_gt, w, h)
            neg_correct = neg_iou > 0.5
        except:
            neg_correct = False
        
        key = ('TP' if pos_correct else 'FP') + ('_TN' if neg_correct else '_FN')
        stats[key] += 1
    
    total = sum(stats.values())
    print(f"正正确 | 负正确: {stats['TP_TN']/total:.2%}")
    print(f"正正确 | 负错误: {stats['TP_FN']/total:.2%}") 
    print(f"正错误 | 负错误: {stats['FP_FN']/total:.2%}")
    print(f"正错误 | 负正确: {stats['FP_TN']/total:.2%}")

if __name__ == "__main__":
    main()