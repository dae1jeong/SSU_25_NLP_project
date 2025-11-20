from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# 인증 라우터
from login.auth.routes import router as auth_router
# RAG 파이프라인 (클래스만 가져오고 실행은 아직 안 함!)
from RAG.rag_pipeline import RAGPipeline, call_openai_api

load_dotenv()

app = FastAPI(title="SSU Chatbot & Auth System")

app.mount("/static", StaticFiles(directory="login/static"), name="static")
templates = Jinja2Templates(directory="login/templates")
app.include_router(auth_router, prefix="/auth")

# 🌟 [핵심 변경] 전역 변수를 None으로 설정 (아직 로딩 안 함)
rag_instance = None

def get_rag_engine():
    """
    RAG 엔진이 필요할 때만 호출되는 함수입니다.
    아직 로딩이 안 되어 있으면 그때 로딩합니다. (Lazy Loading)
    """
    global rag_instance
    if rag_instance is None:
        print("\n💤 [System] RAG 엔진이 아직 잠들어 있습니다. 깨우는 중... (로딩 시작)")
        rag_instance = RAGPipeline() # 이때 DB 읽고 토큰화 하느라 시간이 좀 걸림
        print("☀️ [System] RAG 엔진 로딩 완료! 이제부터 답변이 빠릅니다.\n")
    return rag_instance

# -----------------------------------------------------------
# 페이지 라우터
# -----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/verify", response_class=HTMLResponse)
def verify_page(request: Request, email: str = None):
    return templates.TemplateResponse("verify.html", {"request": request, "email": email})

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    # 사용자가 채팅방에 들어오면 미리 로딩을 시작해두면 좋습니다. (선택사항)
    # 지금은 일단 첫 질문 때 로딩하게 둡니다.
    return templates.TemplateResponse("chat.html", {"request": request})

# -----------------------------------------------------------
# API 라우터
# -----------------------------------------------------------
class ChatRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: ChatRequest):
    user_query = req.question
    print(f"📩 [질문 수신] {user_query}")
    
    # 🌟 여기서 엔진을 가져옵니다. (첫 질문이라면 로딩하느라 시간이 좀 걸림)
    rag = get_rag_engine()
    
    # 답변 생성
    answer = rag.answer_with_llm(user_query, llm_call=call_openai_api)
    
    print(f"📤 [답변 전송] 완료")
    return JSONResponse(content={"answer": answer})