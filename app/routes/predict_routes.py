from fastapi import APIRouter, Depends, HTTPException
from app.utils.security import get_current_user
from app.services.predict_service import predict_heart_disease

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.get("/")
async def predict(current_user: str = Depends(get_current_user)):
    result = await predict_heart_disease(current_user)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Please add profile data first."
        )

    return result