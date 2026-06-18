from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

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
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": file_path
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

class QuestionRequest(BaseModel):
    question: str
    filename: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        return {
            "success": True,
            "answer": f"Dummy answer for: {request.question}",
            "sources": [{"page": 1, "type": "text"}]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/files")
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return {"files": files}