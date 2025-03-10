from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
import torch
from func import eval_model
model_path = "checkpoints/VisCoT-13b-336"

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

prompt = "What sport is the yellow sculpture engaged in?"
#prompt = "What sport is the green sculpture engaged in?"
image_file = "img.jpg,img.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

x = eval_model(args,model,tokenizer,image_processor)
print(x)
