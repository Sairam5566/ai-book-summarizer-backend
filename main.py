from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import uuid
import json
import asyncio
from typing import Dict
import os
from transformers import pipeline
import spacy
import networkx as nx

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-book-summarizer.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

# Store processing status
processing_tasks: Dict[str, dict] = {}
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def extract_key_phrases(text: str) -> list:
    """Extract key phrases from text using spaCy."""
    doc = nlp(text)
    key_phrases = []
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:  # Only phrases with 2 or more words
            key_phrases.append(chunk.text)
    
    return list(set(key_phrases))  # Remove duplicates

def split_into_chunks(text: str, max_chunk_size: int = 1000) -> list:
    """Split text into smaller chunks."""
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence:
        sentences.append(current_sentence.strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def process_book(file_path: Path, task_id: str):
    """Process book text and generate summaries."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into chunks
        chunks = split_into_chunks(text)
        
        # Generate summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            chunk_summaries.append(summary)

        # Generate final summary
        combined_summary = " ".join(chunk_summaries)
        final_summary = summarizer(combined_summary, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        # Extract key phrases
        key_phrases = extract_key_phrases(text)

        # Save results
        result = {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'key_phrases': key_phrases[:20]  # Limit to top 20 key phrases
        }

        result_path = RESULTS_DIR / f"{task_id}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = result

    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a book file and start processing."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    # Generate unique task ID
    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}.txt"

    try:
        # Save uploaded file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        # Initialize task status
        processing_tasks[task_id] = {
            'status': 'processing',
            'filename': file.filename
        }

        # Start processing in background
        asyncio.create_task(process_book(file_path, task_id))

        return {"id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get the processing status and results for a task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_tasks[task_id]
    response = {"status": task["status"]}

    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
