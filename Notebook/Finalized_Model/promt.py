# !pip install -q accelerate==0.25.0 peft==0.4.0 transformers==4.36.2 datasets==2.16.1
# !pip install -q llama-cpp-python

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
ADAPTER_PATH = "Notebook/Finalized_Model/Model"

MERGED_MODEL_DIR = "./merged_model"

GGUF_OUTPUT_PATH = "./pythia-410m-gallstone-assistant.gguf"

print(f"Loading base model: {BASE_MODEL_NAME}")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    return_dict=True,
    torch_dtype=torch.float16, 
)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")

try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Successfully loaded adapter.")
except Exception as e:
    print(f"Error loading adapter: {e}")
    print("Please ensure the ADAPTER_PATH is correct and points to the directory containing 'adapter_model.bin'.")
    exit()


print("Merging adapter weights with the base model...")

merged_model = model.merge_and_unload()

print("Merge complete.")

print(f"Saving merged model to: {MERGED_MODEL_DIR}")

os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

merged_model.save_pretrained(MERGED_MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print("Merged model and tokenizer saved successfully.")

print("\n" + "="*50)
print("Converting to GGUF format...")
print("="*50)

if not os.path.exists("llama.cpp"):
    print("Cloning llama.cpp repository...")
    os.system("git clone https://github.com/ggerganov/llama.cpp.git")
else:
    print("llama.cpp repository already exists.")

conversion_script_path = "llama.cpp/convert.py"

command = (
    f"python {conversion_script_path} {MERGED_MODEL_DIR} "
    f"--outfile {GGUF_OUTPUT_PATH} "
    f"--outtype f16" 
)

print(f"Running conversion command:\n{command}")

os.system(command)

print("\nConversion to GGUF complete!")
print(f"Your GGUF model is saved at: {GGUF_OUTPUT_PATH}")
print("You can now download this file and use it with any GGUF-compatible runner (e.g., LM Studio, llama.cpp).")

