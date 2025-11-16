"""
Standalone script to run a ColPali-based RAG API server using FastAPI.

Example
-------
    python rag_server.py

Description
-----------
This module loads a ColPali multimodal RAG model, exposes endpoints for:

1. `/upload` : Upload and index PDF documents.
2. `/query`  : Query top-K relevant pages from the indexed PDF.

The API server uses:
- FastAPI for routing
- RAGMultiModalModel (Byaldi) for multimodal search
- Uvicorn as the ASGI server

Notes
-----
This script assumes:
- PDFs will be stored in a local `./docs` directory.
- Indexes will be stored in-memory or byaldi’s internal format.

"""

import os
import argparse
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from byaldi import RAGMultiModalModel
import uvicorn


# ---------------------------------------------------------------------------
# Global Model Handle
# ---------------------------------------------------------------------------

RAG_MODEL = None


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Query payload for retrieving top-K similar pages.

    Parameters
    ----------
    query : str
        The natural language query.
    k : int, optional
        Number of top results to return (default = 3).
    """
    query: str
    k: int = 3


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(device: str = "cpu"):
    """
    Load the ColPali RAG model.

    Parameters
    ----------
    device : str
        Device to load the model on ('cpu' or 'cuda').

    Returns
    -------
    RAGMultiModalModel
    """
    print("Loading ColPali model...")
    model = RAGMultiModalModel.from_pretrained(
        "vidore/colpali-v1.2",
        verbose=0,
        device=device
    )
    print("Model loaded!")
    return model


# ---------------------------------------------------------------------------
# PDF Handling & Query Logic
# ---------------------------------------------------------------------------

def save_pdf(file: UploadFile, target_dir: str = "./docs") -> str:
    """
    Save the uploaded PDF to local storage.

    Parameters
    ----------
    file : UploadFile
        Uploaded PDF file.
    target_dir : str
        Directory to store the file.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_path


def index_pdf(model, file_path: str):
    """
    Index a PDF document using the RAG model.

    Parameters
    ----------
    model : RAGMultiModalModel
        Loaded ColPali model.
    file_path : str
        Location of the PDF to index.

    Returns
    -------
    None
    """
    model.index(
        input_path=file_path,
        index_name="default",
        store_collection_with_index=True,
        overwrite=True
    )


def run_query(model, query: str, k: int):
    """
    Execute a semantic search on the indexed document.

    Parameters
    ----------
    model : RAGMultiModalModel
    query : str
    k : int

    Returns
    -------
    list of dict
        List of results containing page number & base64 image.
    """
    results = model.search(query, k=k)

    if not results:
        raise HTTPException(status_code=404, detail="No results found")

    output = []
    for idx, result in enumerate(results):
        output.append(
            {
                "rank": idx + 1,
                "page_num": result.page_num,
                "image_base64": result.base64
            }
        )
    return output


# ---------------------------------------------------------------------------
# FastAPI App Construction
# ---------------------------------------------------------------------------

def build_app():
    """
    Build and configure the FastAPI application.

    Returns
    -------
    FastAPI
    """
    app = FastAPI(title="ColPali RAG API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {"status": "running", "model_loaded": RAG_MODEL is not None}

    # Upload PDF
    @app.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)):
        try:
            file_path = save_pdf(file)
            index_pdf(RAG_MODEL, file_path)
            return {"status": "success", "filename": file.filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Query API
    @app.post("/query")
    async def query_pdf(request: QueryRequest):
        try:
            results = run_query(RAG_MODEL, request.query, request.k)
            return {"query": request.query, "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    """
    Program entry point.
    Loads the model, builds the API app, and starts the server.
    """
    global RAG_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load model on: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on."
    )
    args = parser.parse_args()

    RAG_MODEL = load_model(device=args.device)
    app = build_app()

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
