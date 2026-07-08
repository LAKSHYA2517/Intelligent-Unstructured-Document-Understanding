"""Entrypoint shim so `uvicorn main:app` and `python main.py` keep working
with the FastAPI app now living in the `app` package."""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
