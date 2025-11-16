from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
from byaldi import RAGMultiModalModel
import uvicorn

app = FastAPI(title="ColPali RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_MODEL = None

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.on_event("startup")
async def startup_event():
    global RAG_MODEL
    print("Loading ColPali model...")
    RAG_MODEL = RAGMultiModalModel.from_pretrained(
        "vidore/colpali-v1.2",
        verbose=0,
        device="cpu"
    )
    print("Model loaded!")

@app.get("/")
async def root():
    return {"status": "running", "model_loaded": RAG_MODEL is not None}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    try:
        os.makedirs("./docs", exist_ok=True)
        file_path = f"./docs/{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        RAG_MODEL.index(
            input_path=file_path,
            index_name="default",
            store_collection_with_index=True,
            overwrite=True
        )
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pdf(request: QueryRequest):
    """Query and get top K similar pages as base64 images"""
    try:
        results = RAG_MODEL.search(request.query, k=request.k)
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        pages = []
        for idx, result in enumerate(results):
            pages.append({
                "rank": idx + 1,
                "page_num": result.page_num,
                "image_base64": result.base64
            })
        
        return {"query": request.query, "results": pages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)