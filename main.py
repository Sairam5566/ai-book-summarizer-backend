from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
import requests
from typing import List
import json
import os
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-book-summarizer.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    logger.error("HUGGINGFACE_API_TOKEN not found in environment variables")
    raise ValueError("HUGGINGFACE_API_TOKEN not configured")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def get_summary(text: str) -> str:
    """Get summary using Hugging Face's hosted BART model"""
    try:
        payload = {"inputs": text, "parameters": {"max_length": 300, "min_length": 100}}
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Hugging Face API error: {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error from Hugging Face API: {response.text}"
            )
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]["summary_text"]
        else:
            logger.error(f"Unexpected response format from Hugging Face: {result}")
            raise HTTPException(
                status_code=500,
                detail="Unexpected response format from summarization API"
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error to Hugging Face API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to summarization service"
        )

def extract_entities(text: str) -> List[dict]:
    """Extract named entities using spaCy"""
    try:
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"]:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_
                })
        
        return list({(e["text"], e["type"]): e for e in entities}.values())  # Remove duplicates
    
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing text for entity extraction"
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        logger.info(f"Processing file: {file.filename}")
        
        # Get summary from Hugging Face API
        logger.info("Requesting summary from Hugging Face API")
        summary = get_summary(text)
        
        # Extract entities using spaCy (lightweight)
        logger.info("Extracting entities using spaCy")
        entities = extract_entities(text)
        
        logger.info("Successfully processed file")
        return {
            "summary": summary,
            "entities": entities
        }
        
    except UnicodeDecodeError:
        logger.error(f"Error decoding file: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="File must be a valid UTF-8 encoded text file"
        )
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Verify spaCy model is loaded
        nlp("Test sentence")
        
        # Verify Hugging Face API token is configured
        if not HUGGINGFACE_API_TOKEN:
            raise ValueError("HUGGINGFACE_API_TOKEN not configured")
            
        return {
            "status": "healthy",
            "spacy_model": "loaded",
            "huggingface_token": "configured"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
