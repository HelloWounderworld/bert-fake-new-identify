from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/request_to_bert")
def analyzing_gpu(request: Request):
    ...