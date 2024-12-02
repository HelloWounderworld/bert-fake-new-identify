from fastapi import APIRouter, Request
import json
import aiohttp
import asyncio

router = APIRouter()

@router.post("/requestbert")
def analyzing_gpu(request: Request):
    URL_REQUEST = "http://localhost:8003/check_text/"
    content_type = request.headers.get('Content-Type')
    json_data = await request.json()

    print(json_data)

    data = {
        'text': json_data['text']
    }

    if content_type is None:
        raise HTTPException(status_code=400, detail='No Content-Type provided')
    elif content_type == 'application/json':
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(URL_REQUEST, json=data) as response:
                    result = await response.json()
                    return {
                        "class": result['class'],
                        "fake": result['probabilities']['Fake'],
                        "real": result['probabilities']['Real']
                    }

        except JSONDecodeError:
            raise HTTPException(status_code=400, detail='Invalid JSON data')
        except aiohttp.ClientResponseError as e:
            raise HTTPException(status_code=e.status, detail=f"Error at the external API: {e.message}")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Unexpected error to access external API...")
    else:
        raise HTTPException(status_code=400, detail='Content-Type not supported')
        