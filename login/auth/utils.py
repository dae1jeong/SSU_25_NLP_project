# ==============================================================================
# SSU_25_NLP_project - login/utils.py
#
# [개요]
# 로그인/인증 시스템에서 사용되는 핵심 유틸리티 함수들을 모아둔 파일입니다.
#
# [주요 역할]
# 1. 인증 코드 생성: generate_code()를 통해 6자리 랜덤 인증 번호를 생성합니다.
# 2. 이메일 전송: send_verification_email()을 통해 SendGrid API를 호출하여
#    생성된 인증 코드를 사용자 이메일로 안전하게 전송합니다.
#    - 환경 변수(.env)에서 SendGrid API Key를 로드하여 사용합니다.
# ==============================================================================

import random
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

def generate_code() -> str:
    """6자리 인증번호 생성 (보안 강화를 위해 6자리 추천)"""
    return str(random.randint(100000, 999999))

async def send_verification_email(to_email: str, code: str):
    """SendGrid를 이용해 인증 이메일 전송"""
    
    # 환경 변수에서 값 가져오기
    from_email = os.getenv("FROM_EMAIL")
    api_key = os.getenv("SENDGRID_API_KEY")

    if not api_key or not from_email:
        # 에러가 나면 로그를 찍어서 확인
        print(f"🚫 [Email Error] 키 없음. API_KEY: {bool(api_key)}, FROM: {from_email}")
        raise Exception("SENDGRID_API_KEY 또는 FROM_EMAIL 환경 변수가 설정되지 않았습니다.")

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="[숭실대학교] 재학생 이메일 인증 코드",
        html_content=f"""
            <div style="font-family: sans-serif; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h3 style="color: #007bff;">궁금했슈(SSU) 인증번호</h3>
                <p>아래 6자리 인증번호를 입력해주세요.</p>
                <h1 style="letter-spacing: 5px;">{code}</h1>
                <p style="color: #888; font-size: 12px;">이 번호는 5분간 유효합니다.</p>
            </div>
        """
    )
    
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"📧 [Email Sent] To: {to_email} | Status: {response.status_code}")
    except Exception as e:
        print(f"🔥 [Email Failed] {e}")
        raise e