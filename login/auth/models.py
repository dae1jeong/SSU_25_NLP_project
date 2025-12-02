# ==============================================================================
# SSU_25_NLP_project - login/models.py
#
# [개요]
# FastAPI 서버에서 사용되는 데이터 유효성 검사 모델을 정의합니다.
# 사용자 요청 시 데이터의 형태와 타입을 명확히 규정하여 서버의 안정성을 높입니다.
#
# [주요 모델]
# 1. EmailRequest: 이메일 인증 요청 시 사용 (형식: EmailStr)
# 2. VerifyRequest: 인증 코드 확인 요청 시 사용 (이메일 + 6자리 코드)
# 3. ChatRequest: 챗봇 API (/api/ask) 호출 시 사용 (질문 텍스트)
# ==============================================================================
# 작업자 : 박대정

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