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
import ast
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import cv2
from ..extras import logging
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import ImageInput, VideoInput
    from .parser import DatasetAttr


logger = logging.get_logger(__name__)


def _convert_images(
    images: Union["ImageInput", Sequence["ImageInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["ImageInput"]]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if not isinstance(images, list):
        images = [images]
    elif len(images) == 0:
        return None
    else:
        images = images[:]

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(images)):
            if isinstance(images[i], str) and os.path.isfile(os.path.join(data_args.image_dir, images[i])):
                images[i] = os.path.join(data_args.image_dir, images[i])

    return images


def _convert_videos(
    videos: Union["VideoInput", Sequence["VideoInput"]],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Optional[List["VideoInput"]]:
    r"""
    Optionally concatenates video path to dataset dir when loading from local disk.
    """
    if not isinstance(videos, list):
        videos = [videos]
    elif len(videos) == 0:
        return None
    else:
        videos = videos[:]

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(videos)):
            if isinstance(videos[i], str) and os.path.isfile(os.path.join(data_args.image_dir, videos[i])):
                videos[i] = os.path.join(data_args.image_dir, videos[i])

    return videos


def convert_alpaca(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    prompt = []
    if dataset_attr.history and isinstance(example[dataset_attr.history], list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    query = []
    if dataset_attr.prompt and example[dataset_attr.prompt]:
        query.append(example[dataset_attr.prompt])

    if dataset_attr.query and example[dataset_attr.query]:
        query.append(example[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], str)
        and isinstance(example[dataset_attr.rejected], str)
    ):  # pairwise example
        response = [
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.chosen]},
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.rejected]},
        ]
    elif dataset_attr.response and isinstance(example[dataset_attr.response], str):  # normal example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
    else:  # unsupervised
        response = []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": example[dataset_attr.system] if dataset_attr.system else "",
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def convert_sharegpt_step(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    
    # print('这是每一条数据:')
    # print(example)
    # print('111'*10)
    # print(dataset_attr.messages)
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example["chosen_step"], dict)
        and isinstance(example["rejected_step"], dict)
    ):  # pairwise example
        chosen = example["chosen_step"]
        rejected = example["rejected_step"]
        if (
            chosen[dataset_attr.role_tag] not in accept_tags[-1]
            or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output

def crop_and_convert_to_image(image_path, bbox, output_folder=None):
    """
    裁剪图像并将裁剪后的子图转换为 Base64 编码，同时保存裁剪后的图像到指定文件夹。

    参数:
        image_path (str): 输入图像的路径。
        bbox (list): 归一化的边界框坐标 [x_min, y_min, x_max, y_max]。
        output_folder (str): 保存裁剪后图像的文件夹路径。如果为 None，则不保存图像。

    返回:
        tuple: (base64_image, output_filename) 
            base64_image (str): 裁剪后的图像的 Base64 编码。
            output_filename (str): 保存的文件名，如果 output_folder 不为 None；否则为 None。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 将归一化的坐标转换为实际像素坐标
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    # 裁剪图像
    cropped_image = image[y_min:y_max, x_min:x_max]

    output_filename = None
    # 如果指定了输出文件夹，保存裁剪后的图像
    if output_folder is not None:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成唯一的文件名
        base_name = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(base_name)
        # 将 bbox 归一化坐标格式化为字符串，保留三位小数
        bbox_str = '_'.join([f"{coord:.3f}".replace('.', '_')  for coord in bbox])
        # 组合新文件名
        output_filename = f"{name_part}_{bbox_str}{ext_part}"
        output_path = os.path.join(output_folder, output_filename)
        
        # 保存裁剪后的图像
        cv2.imwrite(output_path, cropped_image)
    return output_path

def convert_sharegpt(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    
    # print('这是每一条数据:')
    # print(example)
    # print('111'*10)
    # print(dataset_attr.messages)
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""

    aligned_messages = []
    prompt_step = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True
        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
        )
        prompt_step.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag] + '\n<image>'}
        )
    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        chosen_step = example["chosen_step"]
        rejected_step = example["rejected_step"]
        if (
            chosen[dataset_attr.role_tag] not in accept_tags[-1]
            or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
        ]
        response_step = [
            {"role": tag_mapping[chosen_step[dataset_attr.role_tag]], "content": chosen_step[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected_step[dataset_attr.role_tag]], "content": rejected_step[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)


    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    ancestor_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    prefix_path = os.path.join(ancestor_dir, 'data/')
    output_dir = os.path.join(ancestor_dir, 'image_crop_tmp/')

    # print(prefix_path)
    # ### 将crop的图片保存到本地路径下
    # prefix_path = '/datadisk/zhangquan/LLaMA-Factory/data/'
    # output_dir = '/datadisk/zhangquan/LLaMA-Factory/image_crop_tmp/'
    crop_image = crop_and_convert_to_image(prefix_path + example[dataset_attr.images][0], ast.literal_eval(chosen_step[dataset_attr.content_tag]),output_dir)
    image_step = [example[dataset_attr.images][0], crop_image]
    output = {
        "_prompt_step1": prompt,
        "_prompt_step2": prompt_step,
        "_response_step1": response_step,
        "_response_step2": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images_step1": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_images_step2": convert_images(image_step) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments"
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        # if step == True:
        #     print(111)
        #     convert_func = partial(convert_sharegpt_step, dataset_attr=dataset_attr, data_args=data_args)
        # else:
        #     print(222)
        #     convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)
            
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
