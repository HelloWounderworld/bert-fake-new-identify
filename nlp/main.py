from fastapi import FastAPI, HTTPException
from fastapi.middleware import CORSMiddleware

from check.request import router as check_text
from testing_gpu.request import router as test_cuda_gpu_available
from accuracy.request import router as evaluate_accuracy

app = FastAPI()

app.include_router(check_text)
app.include_router(test_cuda_gpu_available)
app.include_router(evaluate_accuracy)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
