### self evolution to generate VQA data with bounding box

from openai import OpenAI
from transformers.utils.versions import require_version
import base64
import json
import ast
import cv2
import base64
from openai import AzureOpenAI
from datetime import date
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from torchvision.ops import box_iou
from PIL import Image
import torch
import random

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def generate_random_bounding_box(image_width, image_height, standard_bbox, max_size_diff=0.3, max_attempts=1000):
    x_min_std, y_min_std, x_max_std, y_max_std = standard_bbox
    std_width = x_max_std - x_min_std
    std_height = y_max_std - y_min_std

    for i in range(max_attempts):
        if i % 20 == 0:
            max_size_diff = max_size_diff + 0.1
        min_width = std_width * (1 - max_size_diff)
        max_width = std_width * (1 + max_size_diff)
        min_height = std_height * (1 - max_size_diff)
        max_height = std_height * (1 + max_size_diff)
        width = random.uniform(min_width, max_width)
        height = random.uniform(min_height, max_height)
        x_min = random.uniform(0, 1 - width)
        y_min = random.uniform(0, 1 - height)
        x_max = x_min + width
        y_max = y_min + height

        no_overlap = (x_max <= x_min_std or
                      x_min >= x_max_std or
                      y_max <= y_min_std or
                      y_min >= y_max_std)

        if no_overlap:
            new_bbox_normalized = [x_min, y_min, x_max, y_max]
            return new_bbox_normalized

    raise ValueError("Does not find boundingbox")

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

# lock
write_lock = threading.Lock()

def crop_and_convert_to_base64(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        print(f"error: can not load image {image_path}")
        return None

    height, width = image.shape[:2]
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    cropped_image = image[y_min:y_max, x_min:x_max]
    _, buffer = cv2.imencode('.jpg', cropped_image)
    return base64.b64encode(buffer).decode('utf-8')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_result(base64_image,prompt,client,client_judge,image_path,gt_response,boundingbox_good,bad_generate):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]
    }]
    
    result = client.chat.completions.create(
        messages=messages, 
        model="test",
        temperature=2
    )
    boundingbox_response = result.choices[0].message.content
    if bad_generate:
        try:
            with Image.open(image_path[0]) as img:
                w, h = img.size
            pos_box_bad = json.loads(boundingbox_response)
            pos_box_good = json.loads(boundingbox_good)
            pos_iou = convert_and_compute_iou(pos_box_good, pos_box_bad, w, h)
            if pos_iou > 0.05:
                bbox = generate_random_bounding_box(w, h, ast.literal_eval(boundingbox_good), max_size_diff=0.3, max_attempts=100)
            else:
                bbox = ast.literal_eval(boundingbox_response)
            boundingbox_response = str(bbox)
            crop_base64_image = crop_and_convert_to_base64(image_path[0], bbox)
            if not crop_base64_image:
                return None
        except Exception as e:
            print(f"fail to create boundingbix: {e}")
            return None
    else:
        try:
            bbox = ast.literal_eval(boundingbox_response)
            crop_base64_image = crop_and_convert_to_base64(image_path[0], bbox)
            if not crop_base64_image:
                return None
        except Exception as e:
            print(f"boundingbox error: {e}")
            return None

    messages_new = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt + '\n<image>'},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_base64_image}"}}
        ]
    }]
    
    result_new = client.chat.completions.create(
        messages=messages_new, 
        model="test",
        temperature=2
    )
    response = result_new.choices[0].message.content

    gt_questin = prompt.replace('<image>\n','').replace('Please first provide the bounding box coordinate of the region, then refer to the corresponding sub-image to answer the question better.','')
    judge_score_prompt = f'''You are responsible for proofreading the answers, you need to give the score to the model's answer by referring to the standard answer, based on the given question and image.
    The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
    The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.

    Question: {gt_questin}
    Standard answer: {gt_response}
    Model's answer: {response}
    '''
    
    judge_messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": judge_score_prompt}
        ]
    }]

    result = client_judge.chat.completions.create(
        messages=judge_messages, 
        model="test",
        temperature=0
    )
    judge_score = result.choices[0].message.content
    
    accuracy = None

    try:
        score_data = json.loads(judge_score)
        accuracy = score_data["score"]
    except:
        try:
            clean_score = re.sub(r'```json|```', '', judge_score).strip()
            score_data = json.loads(clean_score)
            if isinstance(score_data, float) or isinstance(score_data, int):
                accuracy = score_data
            else:
                accuracy = score_data["score"]
        except Exception as e:
            print(judge_score)
            print(f"response fail: {e}")

    judge_bbox_prompt = f'''You are responsible for verifying the relevance of the image based on the provided question and standard answer, you need to assess whether the image aligns with the standard answer.
    The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
    The evaluation criteria is that, the higher score will be if the image effectively encompasses the information provided in the standard answer based on question.

    Question: {gt_questin}
    Standard answer: {gt_response}
    '''
    
    judge_bbox_messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_base64_image}"}},
            {"type": "text", "text": judge_bbox_prompt}
        ]
    }]

    result_bbox = client_judge.chat.completions.create(
        messages=judge_bbox_messages, 
        model="test",
        temperature=0
    )
    judge_bbox_score = result_bbox.choices[0].message.content
    
    accuracy_bbox = None

    try:
        score_bbox_data = json.loads(judge_bbox_score)
        accuracy_bbox = score_bbox_data["score"]
    except:
        try:
            clean_bbox_score = re.sub(r'```json|```', '', judge_bbox_score).strip()
            score_bbox_data = json.loads(clean_bbox_score)
            if isinstance(score_bbox_data, float) or isinstance(score_bbox_data, int):
                accuracy_bbox = score_bbox_data
            else:
                accuracy_bbox = score_bbox_data["score"]
        except Exception as e:
            print(judge_bbox_score)
            print(f"bbox fail: {e}")
    return accuracy,accuracy_bbox,boundingbox_response,response
                     
def process_single_attempt(image_path, prompt, gt_response, gt_bbox, base64_image):
    try:
        prompt = prompt.replace('Please provide the bounding box coordinate of the region that can help you answer the question better.','Please first provide the bounding box coordinate of the region, then refer to the corresponding sub-image to answer the question better.')
        client = OpenAI(
            api_key=os.environ.get("API_KEY", "0"),
            base_url=f"http://localhost:{os.environ.get('API_PORT', 8010)}/v1",
        )
        client_judge = OpenAI(
            api_key=os.environ.get("API_KEY", "0"),
            base_url=f"http://localhost:{os.environ.get('API_PORT', 8000)}/v1",
        )
        if process_result(base64_image,prompt,client,client_judge,image_path,gt_response,None,False) == None:
            return None
        else:   
            accuracy1,accuracy_bbox1,boundingbox_response1,response1 = process_result(base64_image,prompt,client,client_judge,image_path,gt_response,None,False)
            
        if process_result(base64_image,prompt,client,client_judge,image_path,gt_response,boundingbox_response1,True) == None:
            return None
        else:   
            accuracy2,accuracy_bbox2,boundingbox_response2,response2 = process_result(base64_image,prompt,client,client_judge,image_path,gt_response,boundingbox_response1,True)
            
        return [{
            'accuracy': accuracy1,
            'accuracy_bbox': accuracy_bbox1,
            'prompt': prompt,
            'image': image_path[0],
            'bbox': boundingbox_response1,
            'gt_bbox': gt_bbox,
            'response': response1,
            'gt_response': gt_response,
            },
            {
            'accuracy': accuracy2,
            'accuracy_bbox': accuracy_bbox2,
            'prompt': prompt,
            'image': image_path[0],
            'bbox': boundingbox_response2,
            'gt_bbox': gt_bbox,
            'response': response2,
            'gt_response': gt_response,
        }]
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None

def process_item(item, num, output_file):
    result_arr = []
    image_path = item['image']
    print(f'Processing data #{num}: {image_path[0]}')
    
    base64_image = encode_image(image_path[0])
    prompt = item['conversations'][0]['value']
    gt_response = item['conversations'][1]['value']
    gt_bbox = item['bbox']

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(
            process_single_attempt,
            image_path,
            prompt,
            gt_response,
            gt_bbox,
            base64_image
        ) for _ in range(3)]
        
        for future in futures:
            try:
                result = future.result()
                if result:
                    result_arr.extend(result)
            except Exception as e:
                print(f"fail to deal: {e}")

    with write_lock:
        if result_arr == []:
            pass
        else:
            with open(output_file, 'a') as outfile:
                if outfile.tell() == 0:
                    outfile.write('[\n')
                else:
                    outfile.write(',\n')
                json.dump(result_arr, outfile, indent=4)
                outfile.flush()
    return num

def main():
    # init the file
    input_file = 'dpo_dataset_slice2.json'
    output_file = 'dpo_dataset_slice2_self_evolution.json'
    with open(output_file, 'w') as f:
        pass

    with open(input_file, 'r') as file:
        data = json.load(file)

    with ThreadPoolExecutor(max_workers=40) as outer_executor:
        futures = [outer_executor.submit(process_item, item, num,output_file) 
                 for num, item in enumerate(data)]
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"fail: {e}")

    with open(output_file, 'a') as f:
        f.write('\n]')

    print("Finish!")

if __name__ == "__main__":
    main()