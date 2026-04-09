from pydantic import BaseModel, EmailStr
from typing import Optional

# ✅ Request model (Signup input)
class User(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    phone: str
    address: Optional[str] = None


# ✅ Stored in DB (with hashed password)
class UserInDB(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: str
    address: Optional[str] = None
    hashed_password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str