# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model, PeftModel


# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name, use_fast=True,
#     cache_dir="./data/model_and_tokenizers"
# )
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # load merged model directly
# base = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.float16, device_map="auto", cache_dir="./data/model_and_tokenizers"
# )
# ft = PeftModel.from_pretrained(base, "./data/deepseek-lora-fixed")
# ft = ft.merge_and_unload()  
# ft.eval()

# def chat_infer(user_text: str, max_new_tokens: int = 128) -> str:
#     messages = [{"role": "user", "content": user_text}]

#     inputs = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#     ).to(ft.device)

#     with torch.inference_mode():
#         out = ft.generate(
#             input_ids=inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=0.8,
#             top_p=0.9,
#             do_sample=True,
#             repetition_penalty=1.15,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     response = tokenizer.decode(
#         out[0][inputs.shape[-1]:], skip_special_tokens=True
#     )
#     return response


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check if running in Docker or locally
if os.path.exists("/deepseek_finetuned_api/API/data/merged_model"):
    model_path = "/deepseek_finetuned_api/API/data/merged_model"
else:
    model_path = "./data/merged_model"

print(f"Loading model from: {model_path}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set DeepSeek/Qwen chat template if missing
if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n<|assistant|>\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"
    print(" Chat template was missing, set manually")

# Load merged model
ft = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Move to device if CPU
if device == "cpu":
    ft = ft.to(device)

ft.eval()

print(" Model loaded successfully!")

def chat_infer(user_text: str, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": user_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(ft.device)
    
    with torch.inference_mode():
        out = ft.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        out[0][inputs.shape[-1]:], skip_special_tokens=True
    )
    return response