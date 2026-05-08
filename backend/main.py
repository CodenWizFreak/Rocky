"""
FastAPI Backend — Rocky

Movie recommendation API with knowledge-graph LightGCN inference and selective forgetting.
(Project Hail Mary energy: one careful pass at the right answer.)

Endpoints:
  GET  /users                         → list of users
  GET  /history/{user_id}             → watch history
  GET  /recommend/{user_id}?k=10      → top-K recommendations
  POST /unlearn                       → trigger unlearning
  GET  /explain/{user_id}/{movie_id}  → explanation for recommendation
  GET  /graph/{user_id}               → subgraph data for D3 visualisation
  GET  /movies/search?q=...           → search movies by title
  GET  /health                        → health check
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import recommend, unlearn, history

app = FastAPI(
    title="Rocky",
    description="Graph-powered MovieLens recommendations with selective forgetting — your science-minded co-pilot for what to watch next.",
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommend.router, tags=["Recommendations"])
app.include_router(unlearn.router, tags=["Unlearning"])
app.include_router(history.router, tags=["Users & History"])


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "rocky",
        "systems": "nominal",
    }


@app.on_event("startup")
async def startup_event():
    """Pre-load data on startup for faster first request."""
    from recommend.inference import get_engine
    try:
        engine = get_engine()
        engine.load()
        print("[Rocky] Recommendation engine loaded on startup")
    except Exception as e:
        print(f"[Rocky] Engine not loaded on startup (train models first): {e}")
