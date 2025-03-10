from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
import torch

def init_viscot(model_path):
    kwargs = {"device_map": "auto",
            # "offload_folder": model_path,
            "cache_dir": r'./'
            }
    kwargs['torch_dtype'] = torch.float16
    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model,tokenizer,image_processor
