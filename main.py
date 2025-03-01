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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-book-summarizer.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models only when needed
summarizer = None
nlp = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    return summarizer

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp

# Store processing status
processing_tasks: Dict[str, dict] = {}
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Read the text content
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Get or initialize models
        summarizer = get_summarizer()
        nlp = get_nlp()

        # Generate summary in chunks to save memory
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        summaries = []
        
        for chunk in chunks:
            if len(chunk.strip()) > 100:  # Only summarize chunks with substantial content
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        final_summary = " ".join(summaries)

        # Process with spaCy for visualization
        doc = nlp(final_summary)
        
        # Extract entities and relationships
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_
            })

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "summary": final_summary,
            "entities": entities
        }

    except Exception as e:
        return {"error": str(e)}

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
