import os
import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ------------------------
# Config - Hugging Face repo
# ------------------------
REPO_ID = "codeXBhawesh/gallstone_chatbot"
FILENAME = "pythia-410m-gallstone-assistant.gguf"

# ------------------------
# Download model from HF Hub
# ------------------------
def get_model_path():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model from Hugging Face: {e}")

MODEL_PATH = get_model_path()

# ------------------------
# Load Model once
# ------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=0,  # change if GPU available
    verbose=True
)

# ------------------------
# Utility functions
# ------------------------
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
    text = re.sub(
        r"between\s+normal\s*\(.*?\)\s+and\s+normal\s*\(.*?\)", 
        "within borderline ranges", text
    )

    # --- Remove hallucinated organs unrelated to gallstones ---
    sentences = text.split(". ")
    sentences = [s for s in sentences if not re.search(r"(pancreas|bowel|liver cyst)", s, re.I)]
    text = ". ".join(sentences)

    # --- Final cleanup ---
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


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

# ------------------------
# Chat function
# ------------------------
def get_chat_response(user_message: str) -> str:
    """Generate a clean response from the model."""
    prompt = create_prompt(user_message)

    output = llm(
        prompt,
        max_tokens=512,
        stop=["### Instruction:", "You:", "### Response:"],
        echo=False
    )

    raw_response = output["choices"][0]["text"]

    # Cleanup pipeline
    response_text = clean_response(raw_response)
    response_text = normalize_medical_terms(response_text)

    if not response_text.strip():
        response_text = "I’m sorry, I couldn’t generate a reliable response. Please try rephrasing your question."

    return response_text
