from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
import requests
from typing import List
import json
import os
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-book-summarizer.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def get_summary(text: str) -> str:
    """Get summary using Hugging Face's hosted BART model"""
    payload = {"inputs": text, "parameters": {"max_length": 300, "min_length": 100}}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error getting summary from Hugging Face API")
    
    return response.json()[0]["summary_text"]

def extract_entities(text: str) -> List[dict]:
    """Extract named entities using spaCy"""
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"]:
            entities.append({
                "text": ent.text,
                "type": ent.label_
            })
    
    return list({(e["text"], e["type"]): e for e in entities}.values())  # Remove duplicates

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        # Get summary from Hugging Face API
        summary = get_summary(text)
        
        # Extract entities using spaCy (lightweight)
        entities = extract_entities(text)
        
        return {
            "summary": summary,
            "entities": entities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
