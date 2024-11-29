from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from requestbert.requestbert import router as request_to_bert

app = FastAPI

app.include_router(request_to_bert)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
