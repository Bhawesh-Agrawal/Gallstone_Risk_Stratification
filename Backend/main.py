from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.chat import get_chat_response
from fastapi.middleware.cors import CORSMiddleware
from api.train import train_rf_pipeline
from api.predict import predict_gallstone
from api.chat_gemini import GeminiChat
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")

# Initialize GeminiChat
gemini = GeminiChat(API_KEY)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
def train_model():
    try:
        result = train_rf_pipeline()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(input_data: dict):
    try:
        result = predict_gallstone(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_text = get_chat_response(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/gemini-chat")
def chat_endpoint(request: ChatRequest):
    try:
        response = gemini.get_response(request.message)
        if response.startswith("ERROR:"):
            raise HTTPException(status_code=500, detail=response)
        return {"prompt": request.message, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Gallstone RF Trainer API is running 🚀"}
