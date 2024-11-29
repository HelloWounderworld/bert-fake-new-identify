from fastapi import APIRouter

router = APIRouter()

@router.get("/evaluate_accurary")
def analyzing_gpu():
    from accuracy.accuracy import test_accuracy

    return test_accuracy()