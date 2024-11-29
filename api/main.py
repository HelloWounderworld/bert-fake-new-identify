from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from accuracy.request import router as evaluate_accurary

app = FastAPI
app.include_router(evaluate_accurary)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
