from fastapi import APIRouter

router = APIRouter()

@router.get("/evaluate_accuracy")
def evaluate_accuracy():
    from accuracy.accuracy import test_accuracy

    return test_accuracy()