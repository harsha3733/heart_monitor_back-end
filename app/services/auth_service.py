from app.db.database import get_db
from app.utils.security import hash_password, verify_password, create_access_token

async def register_user(user):
    db = get_db()

    existing = await db.users.find_one({"email": user.email})
    if existing:
        return None

    user_dict = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "phone": user.phone,
        "address": user.address,
        "hashed_password": hash_password(user.password)
    }

    await db.users.insert_one(user_dict)

    return {
        "email": user.email,
        "message": "User registered successfully"
    }


async def login_user(user):
    db = get_db()

    existing = await db.users.find_one({"email": user.email})
    if not existing:
        return None

    if not verify_password(user.password, existing["hashed_password"]):
        return None

    token = create_access_token({"sub": existing["email"]})

    return {
        "access_token": token,
        "token_type": "bearer"
    }