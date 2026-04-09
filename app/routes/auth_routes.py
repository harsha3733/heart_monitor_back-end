from fastapi import APIRouter, HTTPException
from app.models.user_model import User, UserLogin
from app.services.auth_service import register_user, login_user

router = APIRouter(prefix="/auth", tags=["Auth"])


# 🔹 Register
@router.post("/register")
async def register(user: User):
    result = await register_user(user)

    if not result:
        raise HTTPException(status_code=400, detail="User already exists")

    return result


# 🔹 Login
@router.post("/login")
async def login(user: UserLogin):
    result = await login_user(user)

    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return result