from fastapi import APIRouter, HTTPException, status, Form
from fastapi.responses import RedirectResponse
from motor.motor_asyncio import AsyncIOMotorClient
import os, datetime
from dotenv import load_dotenv
from pathlib import Path

# .models와 .utils에서 필요한 것만 가져옴 (get_hybrid_answer 제거됨)
from .models import EmailRequest, VerifyRequest, ChatRequest
from .utils import generate_code, send_verification_email

# 1. .env 파일 경로 강제 지정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

router = APIRouter()

# 2. DB 연결
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("⚠️ [Auth] MONGO_URI가 없습니다. 로컬 DB를 시도합니다.")
    client = AsyncIOMotorClient("mongodb://localhost:27017")
else:
    client = AsyncIOMotorClient(MONGO_URI)

db = client["ssu_login"]
collection = db["verifications"]

# --- 이메일 인증 코드 발송 ---
@router.post("/send-code")
async def send_code(email: str = Form(...)): 
    print(f"📨 [Auth] 인증 요청 수신: {email}") # 로그 추가
    code = generate_code()
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    
    await collection.update_one(
        {"email": email},
        {"$set": {"code": code, "created_at": datetime.datetime.utcnow(), "expires_at": expiration_time}},
        upsert=True
    )
    
    try:
        await send_verification_email(email, code)
    except Exception as e:
        print(f"❌ [Auth] 이메일 전송 에러: {e}")
        raise HTTPException(status_code=500, detail="이메일 전송 실패")
        
    return RedirectResponse(f"/verify?email={email}", status_code=status.HTTP_303_SEE_OTHER)

# --- 인증 코드 확인 ---
@router.post("/verify-code")
async def verify_code(email: str = Form(...), code: str = Form(...)):
    print(f"🔐 [Auth] 코드 확인 요청: {email} / {code}")
    record = await collection.find_one({"email": email})
    
    if not record or record.get("code") != code:
        raise HTTPException(status_code=401, detail="인증번호가 올바르지 않습니다.")
    
    if record.get("expires_at") and record["expires_at"] < datetime.datetime.utcnow():
        await collection.delete_one({"email": email})
        raise HTTPException(status_code=401, detail="인증번호 만료")

    await collection.delete_one({"email": email})

    print("✅ [Auth] 인증 성공! 채팅방으로 이동합니다.")
    return RedirectResponse("/chat", status_code=status.HTTP_303_SEE_OTHER)