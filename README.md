# AI Book Summarizer Backend

Backend API for the AI Book Summarizer project that processes books and generates summaries using AI.

## Features
- File upload handling
- Text processing and chunking
- AI-powered summarization
- Key phrase extraction
- RESTful API endpoints

## API Endpoints
- `POST /upload` - Upload a book file
- `GET /status/{task_id}` - Get processing status and results

## Deployment
This backend is designed to be deployed on Heroku, Render, or similar platforms.

### Deploy to Heroku
1. Fork this repository
2. Sign up on [Heroku](https://www.heroku.com/)
3. Create a new app
4. Connect to your GitHub repository
5. Deploy settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Deploy to Render
1. Fork this repository
2. Sign up on [Render](https://render.com/)
3. Create a new Web Service
4. Connect to your GitHub repository
5. Deploy settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Local Development
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Technologies Used
- Python 3.8+
- FastAPI
- Hugging Face Transformers
- spaCy
- NetworkX
