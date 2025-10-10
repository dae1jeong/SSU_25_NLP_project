from login.auth.routes import router as auth_router
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="SSU Email Auth System")

# 정적 파일, 템플릿 설정
app.mount("/static", StaticFiles(directory="login/static"), name="static")
templates = Jinja2Templates(directory="login/templates")

# 인증 라우터 등록
app.include_router(auth_router, prefix="/auth")

# 메인 페이지 (이메일 입력)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 인증번호 입력 페이지
@app.get("/verify", response_class=HTMLResponse)
def verify_page(request: Request, email: str):
    return templates.TemplateResponse("verify.html", {"request": request, "email": email})
