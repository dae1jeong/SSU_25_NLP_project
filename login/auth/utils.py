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
        raise Exception("SENDGRID_API_KEY 또는 FROM_EMAIL 환경 변수가 설정되지 않았습니다.")

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="[숭실대학교] 재학생 이메일 인증 코드",
        html_content=f"""
            <h3>안녕하세요. 숭실대학교 재학생 인증 서비스입니다.</h3>
            <p>아래 6자리 인증번호를 웹사이트에 입력하여 인증을 완료해 주세요.</p>
            <p style="font-size: 24px; font-weight: bold; color: #007bff;">인증번호: {code}</p>
            <p>인증번호는 5분간 유효합니다.</p>
        """
    )
    
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        # print(f"SendGrid Status Code: {response.status_code}")
    except Exception as e:
        # 이메일 발송 실패 시 디버깅을 위해 예외를 출력합니다.
        print(f"SendGrid 발송 오류: {e}")
        raise