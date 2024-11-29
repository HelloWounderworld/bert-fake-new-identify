from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from testing_gpu.request import router as test_cuda_gpu_available

app = FastAPI
app.include_router(test_cuda_gpu_available)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
