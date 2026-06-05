"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from audiobench.api.routes import chat, history, settings, transcribe

app = FastAPI(
    title="AudioBench API", version="2.0.0", description="Offline AI audio toolkit — REST API"
)

# Allow CORS for Vite frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcribe"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}
