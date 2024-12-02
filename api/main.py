from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from requestbert.requestbert import router as requestbert

app = FastAPI

app.include_router(requestbert)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
