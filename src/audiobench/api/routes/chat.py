import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import litellm

from audiobench.core.settings import get_settings

router = APIRouter()

@router.post("/")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint powered by LiteLLM.
    
    This is designed to be directly compatible with the Vercel AI SDK 
    (useChat hook) in the frontend.
    """
    body = await request.json()
    settings = get_settings()
    
    # Determine model. If UI doesn't send one, fallback to settings
    model = body.get("model", settings.ollama_model)
    
    # If the user is using ollama, litellm requires the "ollama/" prefix
    if "ollama" in settings.ollama_base_url and not model.startswith("ollama/") and not model.startswith("gemini/"):
        model = f"ollama/{model}"
        
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.3)
    
    # Call LiteLLM asynchronously
    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            api_base=settings.ollama_base_url if model.startswith("ollama/") else None,
            api_key=settings.gemini_api_key if model.startswith("gemini/") else None,
            temperature=temperature,
            stream=True
        )
    except Exception as e:
        # Return a stream error
        async def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate():
        try:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # 0: indicates a text chunk in Vercel Data Stream Protocol
                    yield f'0:{json.dumps(content)}\n'
        except Exception as e:
            # 3: indicates an error chunk
            yield f'3:{json.dumps(str(e))}\n'
            
        # e: indicates finish
        yield 'e:{"finishReason":"stop"}\n'
        
    return StreamingResponse(
        generate(), 
        media_type="text/plain; charset=utf-8",
        headers={"x-vercel-ai-data-stream": "v1"}
    )
