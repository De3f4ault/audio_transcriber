from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from audiobench.core.settings import get_settings, AudioBenchSettings

router = APIRouter()

@router.get("/")
def read_settings() -> dict:
    """Get all current settings."""
    settings = get_settings()
    return settings.model_dump()

@router.patch("/")
def update_settings(updates: Dict[str, Any]) -> dict:
    """Update settings and save to settings.json."""
    settings = get_settings()
    data = settings.model_dump()
    data.update(updates)
    
    try:
        updated_settings = AudioBenchSettings(**data)
        updated_settings.save()
        
        # Update the singleton in-place
        for k, v in updates.items():
            if hasattr(settings, k):
                setattr(settings, k, getattr(updated_settings, k))
                
        return updated_settings.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
