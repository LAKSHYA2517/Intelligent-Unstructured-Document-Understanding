from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

from parser import parse_pdf
from embedder import store_chunks
from answerer import get_answer

app = FastAPI(title="Dell Document QA System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Dell Document QA API is running ✅"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Step 1: Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"✅ File saved: {file.filename}")

        # Step 2: Parse PDF
        print("📄 Parsing PDF...")
        chunks = parse_pdf(file_path)
        print(f"✅ Extracted {len(chunks)} chunks")

        # Step 3: Store in ChromaDB
        print("💾 Storing in ChromaDB...")
        store_chunks(chunks, file.filename)

        return {
            "success": True,
            "message": f"File '{file.filename}' processed successfully",
            "chunks_extracted": len(chunks)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


class QuestionRequest(BaseModel):
    question: str
    filename: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = get_answer(request.question, request.filename)
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/files")
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    pdf_files = [f for f in files if f.endswith(".pdf")]
    return {"files": pdf_files}