from login.auth.routes import router as auth_router
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

app = FastAPI(title="SSU Email Auth System")

# 정적 파일 마운트 (login/static 디렉토리가 있어야 함)
app.mount("/static", StaticFiles(directory="login/static"), name="static")

# 템플릿 설정 (login/templates 디렉토리가 있어야 함)
templates = Jinja2Templates(directory="login/templates")

# 인증 라우터 등록
app.include_router(auth_router, prefix="/auth")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """이메일 입력 메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/verify", response_class=HTMLResponse)
def verify_page(request: Request, email: str = None):
    """인증번호 입력 페이지"""
    # email 파라미터가 없으면 메인 페이지로 리다이렉션하는 로직 추가 가능
    return templates.TemplateResponse("verify.html", {"request": request, "email": email})

# 앱 실행: uvicorn main:app --reload