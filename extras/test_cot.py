# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
from openai import OpenAI
from transformers.utils.versions import require_version
import base64
import ast

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

import cv2

def draw_and_crop_image(image_path, bbox, output_image_with_box, output_cropped_image):
    """
    在原图上绘制红色矩形框并保存，同时裁剪对应的子图并保存。

    参数:
        image_path (str): 输入图像的路径。
        bbox (list): 归一化的边界框坐标 [x_min, y_min, x_max, y_max]。
        output_image_with_box (str): 保存带有红色矩形框的图像路径。
        output_cropped_image (str): 保存裁剪后的图像路径。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 将归一化的坐标转换为实际像素坐标
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    # 在原图上绘制红色矩形框
    color = (0, 0, 255)  # 红色 (BGR格式)
    thickness = 2  # 线宽
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # 保存带有红色矩形框的图像
    cv2.imwrite(output_image_with_box, image)

    # 裁剪图像
    cropped_image = image[y_min:y_max, x_min:x_max]

    # 保存裁剪后的图像
    cv2.imwrite(output_cropped_image, cropped_image)
    

prompt_dict = {
    "sculpture": "What sport is the green sculpture engaged in?",
    "podium": "What letters are written on the logo in front of the podium?",
    "bike": "What is the man leaning against the wall next to the bicycle doing?",
    "shedule": "According to the calendar, what is scheduled for October 12th? What is the specific time?",
    "tel": "What is the applicant's phone number?",
}
def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    messages = []
    exp = "sft_1w"
    type = "shedule"
    prompt = prompt_dict[type]
    
    
    image_path = "exp/" + type + ".jpg"
    output_image_with_box = "exp/" + type + "_image_with_box_" + exp + ".jpg"
    output_cropped_image = "exp/" + type + "_cropped_" + exp + ".jpg"
    prompt_bbox =  prompt + "Please provide the bounding box coordinate of the region that can help you answer the question better."
    img = encode_image(image_path)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img}"}
                },
                {"type": "text", "text": prompt_bbox},
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    print("=========================Get Boundingbox=========================")
    print("Question: <image>", prompt_bbox)
    boundingbox = result.choices[0].message.content
    print("Answer:", boundingbox)
    draw_and_crop_image(image_path, ast.literal_eval(boundingbox), output_image_with_box, output_cropped_image)
    print("=========================Get Response=========================")
    img_crop = encode_image(output_cropped_image)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img}"}
                },
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_crop}"}
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    print("Question: <image>", prompt+"<subimage>")
    print("Answer:", result.choices[0].message.content)
    


if __name__ == "__main__":
    main()
