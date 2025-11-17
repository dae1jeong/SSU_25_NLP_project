from fastapi import APIRouter, HTTPException, status, Form
from fastapi.responses import RedirectResponse # 리디렉션을 위해 추가
from .models import EmailRequest, VerifyRequest
from .utils import generate_code, send_verification_email
from motor.motor_asyncio import AsyncIOMotorClient
import os, datetime
from dotenv import load_dotenv # 환경 변수 로드를 확실히 하기 위해 추가
from .models import ChatRequest # ChatRequest가 없으면 빨간줄 뜸
from .utils import get_hybrid_answer

# .env 파일을 로드합니다.
load_dotenv()

# 🌟 1. 라우터 객체 정의 (NameError 해결)
router = APIRouter()

# MongoDB 연결 설정
MONGO_URI = os.getenv("MONGO_URI")
# 환경 변수가 로드되지 않았다면 서버 시작은 이전에 실패했어야 하지만,
# startup complete 메시지를 받은 것으로 보아, 이미 성공적으로 연결되었습니다.
client = AsyncIOMotorClient(MONGO_URI)
db = client["ssu_login"]
collection = db["verifications"]


@router.post("/send-code")
# 🌟 2. HTML 폼 데이터 수신을 위해 Form 의존성 사용
async def send_code(email: str = Form(...)): 
    
    # # 이메일 도메인 검증
    # if not email.endswith("@soongsil.ac.kr"):
    #     raise HTTPException(
    #         status_code=400, 
    #         detail="숭실대 이메일(@soongsil.ac.kr)만 인증 가능합니다."
    #     )

    # 인증 코드 생성 및 5분 만료 시간 설정
    code = generate_code()
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    
    # MongoDB에 코드 저장/업데이트
    await collection.update_one(
        {"email": email},
        {"$set": {
            "code": code, 
            "created_at": datetime.datetime.utcnow(),
            "expires_at": expiration_time
        }},
        upsert=True
    )
    
    # 이메일 전송 (SendGrid)
    try:
        await send_verification_email(email, code)
    except Exception as e:
        # 이메일 전송 실패 시 500 에러 발생
        raise HTTPException(
            status_code=500,
            detail=f"이메일 전송에 실패했습니다. (SendGrid Key/발신자/인증 확인 필요) 오류: {str(e)}"
        )
        
    # 🌟 3. 인증번호 입력 페이지로 리디렉션
    # HTTP 상태 코드 303은 POST 요청 후 GET 요청으로 페이지를 이동할 때 권장됩니다.
    return RedirectResponse(f"/verify?email={email}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/verify-code")
# 🌟 4. HTML 폼 데이터 수신을 위해 email과 code 모두 Form으로 받음
async def verify_code(email: str = Form(...), code: str = Form(...)):
    
    record = await collection.find_one({"email": email})
    
    # 1. 레코드 존재 여부 및 코드 일치 확인
    if not record or record.get("code") != code:
        raise HTTPException(status_code=401, detail="인증번호가 올바르지 않습니다.")
    
    # 2. 만료 시간 확인
    if record.get("expires_at") and record["expires_at"] < datetime.datetime.utcnow():
        # 만료된 레코드는 삭제
        await collection.delete_one({"email": email})
        raise HTTPException(status_code=401, detail="인증번호가 만료되었습니다. 다시 요청해주세요.")

    # 3. 인증 성공: MongoDB에서 해당 레코드 삭제
    await collection.delete_one({"email": email})

    # 4. 최종 목적지(성공 페이지)로 리디렉션
    return RedirectResponse("https://chat.openai.com", status_code=status.HTTP_303_SEE_OTHER)








#챗봇 질문 api
@router.post("/ask")
async def ask_chatbot(request: ChatRequest):
    """
    챗봇에게 질문을 보냅니다. (로그인 필요 없음 - 필요시 추가 가능)
    """
    response = get_hybrid_answer(request.question)
    return {"answer": response}