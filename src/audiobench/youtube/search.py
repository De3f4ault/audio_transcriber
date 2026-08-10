"""YouTube Data API integration for searching and browsing."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.models import YouTubeChannel

logger = get_logger("youtube.search")

class YouTubeAPIError(Exception):
    pass

class SearchStateExpiredError(Exception):
    pass

@dataclass
class VideoResult:
    n: int
    video_id: str
    title: str
    duration_str: str | None
    published_at: str
    description: str


def _get_api_key() -> str:
    key = get_settings().youtube_api_key
    if not key:
        raise ValueError(
            "YouTube API key is missing. Set AUDIOBENCH_YOUTUBE_API_KEY in your .env file.\n"
            "Get one at: console.cloud.google.com -> Enable 'YouTube Data API v3'"
        )
    return key


def resolve_channel(query: str, session: Any, no_cache: bool = False) -> tuple[str, str]:
    """Resolve a channel name to a channel_id. Caches the result in the database."""
    normalized_query = query.strip().lower()

    if not no_cache:
        # 1. Check Cache
        cached = session.query(YouTubeChannel).filter_by(query=normalized_query).first()
        if cached:
            return cached.channel_id, cached.title or query

    # 2. Hit API
    api_key = _get_api_key()
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": 1,
        "key": api_key,
    }

    response = httpx.get(url, params=params, timeout=10.0)
    if response.status_code != 200:
        logger.error(f"YouTube API error: {response.text}")
        raise YouTubeAPIError(f"Failed to resolve channel '{query}'. HTTP {response.status_code}")

    data = response.json()
    if not data.get("items"):
        raise ValueError(f"No channel found for '{query}'")

    item = data["items"][0]
    channel_id = item["snippet"]["channelId"]
    title = item["snippet"]["title"]

    # 3. Cache the result
    channel_record = YouTubeChannel(
        query=normalized_query,
        channel_id=channel_id,
        title=title,
    )
    session.add(channel_record)
    session.commit()

    return channel_id, title


def _parse_duration(pt_duration: str) -> str:
    """Parse ISO 8601 duration (e.g. PT1H2M3S) to HH:MM:SS."""
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", pt_duration)
    if not match:
        return ""
    
    h, m, s = match.groups()
    h = int(h) if h else 0
    m = int(m) if m else 0
    s = int(s) if s else 0
    
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def search_videos(
    query: str, 
    channel_id: str | None = None, 
    max_results: int = 15, 
    progress_callback=None,
    sort: str = "relevance",
    after: str | None = None,
    before: str | None = None
) -> list[VideoResult]:
    """Search for videos via the YouTube Data API."""
    api_key = _get_api_key()
    
    if progress_callback:
        progress_callback("Searching YouTube...")
        
    # Step 1: Search for videos
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": sort,
        "key": api_key,
    }
    if channel_id:
        params["channelId"] = channel_id
    
    def _format_date(d: str) -> str:
        if len(d) == 10:
            return f"{d}T00:00:00Z"
        return d
        
    if after:
        params["publishedAfter"] = _format_date(after)
    if before:
        params["publishedBefore"] = _format_date(before)

    response = httpx.get(search_url, params=params, timeout=10.0)
    if response.status_code != 200:
        raise YouTubeAPIError(f"Search failed. HTTP {response.status_code}: {response.text}")
        
    data = response.json()
    items = data.get("items", [])
    
    if not items:
        return []

    if progress_callback:
        progress_callback("Fetching durations...")

    # Step 2: We need durations, which requires a videos.list call
    video_ids = [item["id"]["videoId"] for item in items]
    
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    v_params = {
        "part": "contentDetails,snippet",
        "id": ",".join(video_ids),
        "key": api_key,
    }
    v_response = httpx.get(videos_url, params=v_params, timeout=10.0)
    
    durations = {}
    if v_response.status_code == 200:
        v_data = v_response.json()
        for v in v_data.get("items", []):
            durations[v["id"]] = _parse_duration(v["contentDetails"]["duration"])

    import html

    results = []
    for i, item in enumerate(items, 1):
        vid = item["id"]["videoId"]
        snippet = item["snippet"]
        published = snippet["publishedAt"].split("T")[0]
        
        results.append(VideoResult(
            n=i,
            video_id=vid,
            title=html.unescape(snippet["title"]),
            duration_str=durations.get(vid, ""),
            published_at=published,
            description=html.unescape(snippet["description"])
        ))
        
    return results


def write_last_search(results: list[VideoResult]) -> None:
    """Save search results to terminal memory."""
    settings = get_settings()
    path = settings.data_dir / "youtube_last_search.json"
    
    data = {
        "created_at": datetime.now(UTC).isoformat(),
        "ttl_seconds": 3600,
        "results": [
            {
                "n": r.n,
                "video_id": r.video_id,
                "title": r.title,
                "duration": r.duration_str,
                "published": r.published_at
            }
            for r in results
        ]
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_search_result(n: int) -> str:
    """Load a video ID from terminal memory. Raises error if expired."""
    settings = get_settings()
    path = settings.data_dir / "youtube_last_search.json"
    
    if not path.exists():
        raise SearchStateExpiredError("No recent search found. Run `audiobench youtube search` first.")
        
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise SearchStateExpiredError("Search cache is corrupted. Run search again.")
            
    created_at = datetime.fromisoformat(data["created_at"])
    ttl = timedelta(seconds=data.get("ttl_seconds", 3600))
    
    if datetime.now(UTC) > created_at + ttl:
        path.unlink(missing_ok=True)
        raise SearchStateExpiredError("Search results have expired (1 hour TTL). Run `audiobench youtube search` again.")
        
    for r in data.get("results", []):
        if r["n"] == n:
            return r["video_id"]
            
    raise ValueError(f"Result #{n} not found in your last search.")
