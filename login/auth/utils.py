import random
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os

def generate_code():
    """4자리 인증번호 생성"""
    return str(random.randint(1000, 9999))

async def send_verification_email(to_email: str, code: str):
    """SendGrid를 이용해 인증 이메일 전송"""
    message = Mail(
        from_email="no-reply@soongsil.ac.kr",
        to_emails=to_email,
        subject="숭실대학교 이메일 인증 코드",
        html_content=f"<h3>인증번호: {code}</h3><p>이 번호를 웹사이트에 입력해주세요.</p>"
    )
    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
    sg.send(message)
