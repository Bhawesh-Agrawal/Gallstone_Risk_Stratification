import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# 1. Configuration
# ------------------------------------------------------
# --- IMPORTANT: SET YOUR MODEL PATH HERE ---
# This should be the folder where you saved the merged model.
# Using a simple, lowercase name like this is safer.
MODEL_PATH = "D:\\Programming\\Capstone Project\\Notebook\\Finalized_Model\\merged_tinyllama_model"


# 2. Load the Merged Model and Tokenizer
# ------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")

# Load the tokenizer from the saved directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the model from the saved directory
# device_map="auto" will automatically use a GPU if available, otherwise CPU.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("\nModel loaded successfully!")

# 3. Create a Text Generation Pipeline
# ------------------------------------------------------
# The pipeline handles all the complexity of tokenization and generation for you.
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("="*50)
print("Starting interactive chat session.")
print("Type your message and press Enter. Type 'exit' or 'quit' to end.")
print("="*50)

# 4. Interactive Chat Loop
# ------------------------------------------------------
while True:
    # Get user input from the terminal
    user_input = input("You: ")

    # Check for exit commands
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break

    # --- CRITICAL STEP: Format the prompt ---
    # The prompt must EXACTLY match the format used during fine-tuning.
    # Your training script used: "### Instruction:\n\"...\"\n\n### Response:"
    prompt = f"### Instruction:\n\"{user_input}\".\n\n### Response:"

    # Generate a response from the model using the pipeline
    # We pass the full prompt to the pipeline
    result = pipe(
        prompt,
        max_new_tokens=512,  # Maximum number of new tokens to generate
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # The pipeline output includes the original prompt. We can print the full text.
    generated_text = result[0]['generated_text']
    
    # Extract only the part of the text that the model generated
    response_only = generated_text[len(prompt):].strip()

    print(f"Bot: {response_only}")

