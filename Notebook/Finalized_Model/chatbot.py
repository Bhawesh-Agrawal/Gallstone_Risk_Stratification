import os
from llama_cpp import Llama
import re

MODEL_PATH = "D:\\Programming\\Capstone Project\\Notebook\\Finalized_Model\\fine_tuned_model\\pythia-410m-gallstone-assistant.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()

print(f"Loading model from: {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=True
)

print("\nModel loaded successfully!")
print("="*50)
print("Starting interactive gallstone specialist chat.")
print("Type your message and press Enter. Type 'exit' or 'quit' to end.")
print("="*50)


def clean_response(text: str) -> str:
    """Ensure output ends at proper sentence boundary and remove trailing incomplete words."""
    text = text.strip()
    match = re.search(r'(.+[.!?])', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def normalize_medical_terms(text: str) -> str:
    """Rigid cleanup for medical report consistency."""
    # --- Units / ranges ---
    text = re.sub(r"nucleotides per liter\s*\[?IU/L\]?", "IU/L", text)
    text = re.sub(r"\bnU/L\b", "IU/L", text)
    text = re.sub(r"\bIU per week\b", "IU/week", text)
    text = re.sub(r"HGB[^.,;]*", "Hemoglobin (HGB): Normal = 13–17 g/dL (males), 12–15 g/dL (females).", text)
    text = re.sub(r"ECW\s*<\s*10%", "Extracellular Water (ECW) below normal range", text)
    text = re.sub(r"ECW\s*\(10%\)", "Extracellular Water (ECW) near lower bound", text)

    # --- Class Labels ---
    text = text.replace("Class 0 (Positive)", "Class 0 (Healthy/Positive)")
    text = text.replace("Class 1 (Positive)", "Class 1 (Negative/Diseased)")
    text = text.replace("positive (healthy)", "Healthy (Positive)")
    text = text.replace("negative (diseased)", "Diseased (Negative)")

    # --- Cleanup repeated "normal" phrases ---
    text = re.sub(r"between\s+normal\s*\(.*?\)\s+and\s+normal\s*\(.*?\)", 
                  "within borderline ranges", text)

    # --- Remove hallucinated organs unrelated to gallstones ---
    sentences = text.split(". ")
    sentences = [s for s in sentences if not re.search(r"(pancreas|bowel|liver cyst)", s, re.I)]
    text = ". ".join(sentences)

    # --- Final cleanup ---
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# Pre-defined reference injection for grounding
REFERENCE_TEXT = """
Reference Ranges:
- Hemoglobin (males): 13–17 g/dL
- Hemoglobin (females): 12–15 g/dL
- Vitamin D: 30–100 IU/L
- Ultrasound is the first-line diagnostic tool for gallstones
- Surgery is indicated if symptoms are recurrent or severe
"""


def create_prompt(user_input: str) -> str:
    """Wrap user input in a specialist instruction + grounding reference."""
    return f"""
### Instruction:
You are a gallstone specialist. Answer questions **accurately**, using real medical guidelines. 
- Be concise and specific.
- Avoid hallucinating facts about unrelated organs or impossible ranges.
- Focus only on gallstones, symptoms, risk factors, and treatment options.
- Include red flags clearly.
- Maintain a friendly and professional tone.

{REFERENCE_TEXT}

User question: {user_input}

### Response:
"""


while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break

    prompt = create_prompt(user_input)

    output = llm(
        prompt,
        max_tokens=1024,
        stop=["### Instruction:", "You:", "### Response:"],
        echo=False
    )

    raw_response = output["choices"][0]["text"]

    # Step 1: cleanup sentence boundaries
    response_text = clean_response(raw_response)

    # Step 2: normalize medical terms, units, ranges
    response_text = normalize_medical_terms(response_text)

    print(f"\nBot: {response_text}\n")
