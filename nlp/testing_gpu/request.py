from fastapi import APIRouter

router = APIRouter()

@router.get("/test_cuda_gpu_available")
def analyzing_gpu():
    from testing_gpu.diagnose import diagnose

    return diagnose()