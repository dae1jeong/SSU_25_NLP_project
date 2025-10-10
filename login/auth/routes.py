from fastapi import APIRouter, HTTPException
from .models import EmailRequest, VerifyRequest
from .utils import generate_code, send_verification_email
from motor.motor_asyncio import AsyncIOMotorClient
import os, datetime

router = APIRouter()

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["ssu_login"]
collection = db["verifications"]

@router.post("/send-code")
async def send_code(request: EmailRequest):
    if not request.email.endswith("@soongsil.ac.kr"):
        raise HTTPException(status_code=400, detail="숭실대 이메일만 가능합니다.")

    code = generate_code()
    await collection.update_one(
        {"email": request.email},
        {"$set": {"code": code, "created_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    await send_verification_email(request.email, code)
    return {"message": "인증번호가 이메일로 전송되었습니다."}

@router.post("/verify-code")
async def verify_code(request: VerifyRequest):
    record = await collection.find_one({"email": request.email})
    if not record or record["code"] != request.code:
        raise HTTPException(status_code=401, detail="인증번호가 올바르지 않습니다.")
    return {
        "message": "승인되었습니다.",
        "redirect_url": "https://chat.openai.com"
    }
