from fastapi import APIRouter, Request
import json

router = APIRouter()

@router.get("/check_text")
async def analyzing_text(request: Request):
    content_type = request.headers.get('Content-Type')
    json_data = await request.json()

    from check.analyze import parsing

    return parsing(json_data['text'])