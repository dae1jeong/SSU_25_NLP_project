# ==============================================================================
# SSU_25_NLP_project - login/auth/routes.py
#
# [개요]
# FastAPI 서버에서 인증(로그인) 관련 API 엔드포인트를 정의하는 라우터 파일입니다.
#
# [주요 역할]
# 1. DB 연결: MongoDB(AsyncIOMotorClient)에 연결하여 인증 코드를 저장/관리합니다.
# 2. 코드 발송 API: '/send-code' 요청을 받아 인증 코드를 생성, DB에 저장하고 이메일로 발송합니다.
#    - 인증 코드는 5분 유효하며, 만료 시간이 설정됩니다.
# 3. 인증 확인 API: '/verify-code' 요청을 받아 사용자가 입력한 코드가 DB의 코드와 일치하는지 확인합니다.
#    - 인증 성공 시 채팅방 페이지('/chat')로 리다이렉트합니다.
# ==============================================================================


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