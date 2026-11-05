# AI College Campus Assistant — GPT-4o mini + Free Local Embeddings + Streamlit

A PDF RAG assistant using FastAPI, LangChain, OpenAI GPT-4o mini for answer generation, free local Hugging Face embeddings, ChromaDB, and Streamlit.

## Components
- PDF parsing: `PyPDFLoader`
- Chunking: `RecursiveCharacterTextSplitter`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings` (runs locally; no embedding API key or per-request embedding charge)
- Vector store: ChromaDB (persistent)
- Answer generation: OpenAI `gpt-4o-mini` (requires a working OpenAI API key and API access)
- Backend: FastAPI
- Frontend: Streamlit chat UI

## What changed
Only the embedding provider changed. GPT-4o mini remains the answer-generation model. PDF text is embedded locally using the Hugging Face Sentence Transformers model. The first run downloads model files from Hugging Face and may take a few minutes; internet is needed for that initial download. Later embedding runs are local.

## 1. OpenAI API key
Create an API key at https://platform.openai.com/api-keys. It is used only for GPT-4o mini answer generation. API usage is billed separately from a ChatGPT subscription. Never paste your key into chat, screenshots, source code, or GitHub.

## 2. Install (Windows CMD)
Python 3.10+ recommended. Open this folder in VS Code.

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set your OpenAI key. Keep these model settings: 

```env
OPENAI_API_KEY=your_actual_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma_gpt4o_local_embeddings
UPLOAD_DIRECTORY=./data/uploads
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_K=4
MIN_RELEVANCE_SCORE=0.25
MAX_UPLOAD_MB=20
```

Do not commit `.env`. `.gitignore` excludes it.

## 3. Run the application (two terminals)

### Terminal 1 — FastAPI backend
```cmd
.venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8001
```
Backend health: http://127.0.0.1:8001/health  
Swagger docs: http://127.0.0.1:8001/docs

### Terminal 2 — Streamlit frontend
Open a second terminal in the same project folder:
```cmd
.venv\Scripts\activate
streamlit run streamlit_app.py
```
Open http://localhost:8501 if it does not open automatically. Keep the FastAPI terminal running.

## 4. Upload and index sample PDFs
Upload PDFs from `sample_docs/` in the Streamlit sidebar:
- `college_regulations.pdf`
- `student_handbook.pdf`
- `hr_policy.pdf`

The first upload downloads the embedding model if it is not already cached. PDF text is chunked, embedded locally, and stored in ChromaDB.

## 5. Ask questions
Example: `What is the attendance requirement for students?`  
The sample college regulations PDF says students must maintain at least 75% attendance in each course. The response includes source filename and page number.

Other tests:
- How many casual leave days are available to eligible employees?
- What is the hostel curfew?
- What is the library fine for overdue books?
- What is the bonafide certificate procedure?
- What is the capital of Japan? (should return no relevant information)

## Migration note
This version uses a NEW Chroma directory `./data/chroma_gpt4o_local_embeddings`. Do not reuse vectors created by Gemini or OpenAI embeddings; embedding models create different vector spaces. Upload and index all PDFs again in this version.

## Troubleshooting
- `OPENAI_API_KEY is missing`: create `.env` from `.env.example` in the project root and add your key.
- `401`: verify the OpenAI key is active.
- `429`: check OpenAI API quota, rate limits, and billing at https://platform.openai.com/usage.
- Hugging Face download error: check internet access and retry; the embedding model downloads on first use.
- Out of memory: close other applications; this project forces CPU inference.


