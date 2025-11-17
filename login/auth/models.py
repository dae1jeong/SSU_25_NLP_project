from pydantic import BaseModel, EmailStr

class EmailRequest(BaseModel):
    """이메일 인증 요청 시 사용"""
    email: EmailStr

class VerifyRequest(BaseModel):
    """인증 코드 확인 요청 시 사용"""
    email: EmailStr
    code: str
#챗봇질문요청모델
class ChatRequest(BaseModel):
    question: str