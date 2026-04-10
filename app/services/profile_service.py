from app.db.database import get_db

async def create_or_update_profile(email: str, profile_data: dict):
    db = get_db()

    existing = await db.profiles.find_one({"email": email})

    if existing:
        await db.profiles.update_one(
            {"email": email},
            {"$set": profile_data}
        )
        return {"message": "Profile updated"}

    profile_data["email"] = email
    await db.profiles.insert_one(profile_data)

    return {"message": "Profile created"}


async def get_profile(email: str):
    db = get_db()

    # 1️⃣ Try health profile
    profile = await db.profiles.find_one({"email": email})

    if profile:
        profile.pop("_id")

        # Also fetch user personal info and merge so frontend gets phone etc.
        user = await db.users.find_one({"email": email})
        if user:
            profile["first_name"] = user.get("first_name", "")
            profile["last_name"]  = user.get("last_name", "")
            profile["phone"]      = user.get("phone", "")

        return {
            "type": "health",
            "data": profile
        }

    # 2️⃣ Fallback → user basic info
    user = await db.users.find_one({"email": email})

    if user:
        user.pop("_id")
        user.pop("hashed_password")
        return {
            "type": "basic",
            "data": user
        }

    return None