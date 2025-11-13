import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading base model...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=True,
    cache_dir="./API/data/model_and_tokenizers"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./API/data/model_and_tokenizers"
)

print("Loading LoRA adapter...")
# Load LoRA adapter
ft = PeftModel.from_pretrained(base, "./API/data/deepseek-lora-fixed")

print("Merging model...")
# Merge and unload
merged_model = ft.merge_and_unload()

print("Saving merged model...")
# Save the merged model
merged_model.save_pretrained("./API/data/merged_model")
tokenizer.save_pretrained("./API/data/merged_model")

print(" Merged model saved successfully to ./API/data/merged_model")