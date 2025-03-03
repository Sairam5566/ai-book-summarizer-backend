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
        # Split text into chunks of 1000 characters
        max_chunk_size = 1000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = []

        for chunk in chunks:
            if len(chunk.strip()) < 50:  # Skip very small chunks
                continue

            payload = {"inputs": chunk, "parameters": {"max_length": 150, "min_length": 50}}
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Hugging Face API error: {response.text}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error from Hugging Face API: {response.text}"
                )
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summaries.append(result[0]["summary_text"])
            else:
                logger.error(f"Unexpected response format from Hugging Face: {result}")
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected response format from summarization API"
                )

        # Combine summaries
        final_summary = " ".join(summaries)
        return final_summary
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error to Hugging Face API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to summarization service"
        )

def extract_entities(text: str) -> List[dict]:
    """Extract named entities using spaCy"""
    try:
        # Process text in chunks to avoid memory issues
        max_chunk_size = 100000  # 100KB chunks
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        all_entities = []
        
        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"]:
                    all_entities.append({
                        "text": ent.text,
                        "type": ent.label_
                    })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity["text"], entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
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
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode file. Please ensure it's a valid text file."
                )
        
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
            detail="File must be a valid text file"
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
            
        # Test Hugging Face API connection
        test_response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json={"inputs": "Test sentence.", "parameters": {"max_length": 50, "min_length": 10}}
        )
        if test_response.status_code != 200:
            raise ValueError(f"Hugging Face API test failed: {test_response.text}")
            
        return {
            "status": "healthy",
            "spacy_model": "loaded",
            "huggingface_api": "connected"
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
