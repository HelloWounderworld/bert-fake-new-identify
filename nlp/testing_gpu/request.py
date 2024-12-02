from fastapi import APIRouter

router = APIRouter()

@router.get("/test_cuda_gpu_available")
def test_cuda_gpu_available():
    from testing_gpu.test_recognize_gpu import test

    return test()