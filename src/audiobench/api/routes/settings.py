from typing import Any

from fastapi import APIRouter, HTTPException

from audiobench.core.settings import AudioBenchSettings, get_settings

router = APIRouter()


@router.get("/")
def read_settings() -> dict:
    """Get all current settings."""
    settings = get_settings()
    return settings.model_dump()


@router.patch("/")
def update_settings(updates: dict[str, Any]) -> dict:
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

        # Notify the daemon to drop its old settings cache
        from audiobench.daemon.client import DaemonClient
        DaemonClient().reload_settings()

        return updated_settings.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
