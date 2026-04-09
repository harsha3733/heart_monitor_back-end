from fastapi import APIRouter, Depends, HTTPException
from app.models.profile_model import Profile
from app.services.profile_service import create_or_update_profile, get_profile
from app.utils.security import get_current_user

router = APIRouter(prefix="/profile", tags=["Profile"])

# ✅ Create / Update Profile
@router.post("/")
async def save_profile(
    profile: Profile,
    current_user: str = Depends(get_current_user)
):
    result = await create_or_update_profile(
        current_user,
        profile.dict()
    )
    return result


# ✅ Get Profile
@router.get("/")
async def fetch_profile(current_user: str = Depends(get_current_user)):
    profile = await get_profile(current_user)

    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    return profile