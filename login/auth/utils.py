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





#벡터 검색, bm25 구현

import chromadb
import pickle
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import os

print("\n--- ⏳ 챗봇 리소스 로딩 중... ---")

# 1. SBERT (벡터 검색용)
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)

    # 2. ChromaDB (벡터 검색용)
    # (경로가 다를 수 있으니 확인 필요: 보통 프로젝트 루트 기준 ./chroma_db)
    VECTOR_DB_PATH = "./chroma_db" 
    if not os.path.exists(VECTOR_DB_PATH) and os.path.exists("../chroma_db"):
        VECTOR_DB_PATH = "../chroma_db"

    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    vector_collection = chroma_client.get_collection(name="ssu_knowledge_base")

    # 3. BM25 (키워드 검색용)
    BM25_PATH = "bm25_data.pkl"
    if not os.path.exists(BM25_PATH) and os.path.exists("../bm25_data.pkl"):
        BM25_PATH = "../bm25_data.pkl"
    elif not os.path.exists(BM25_PATH) and os.path.exists("data/bm25_data.pkl"):
        BM25_PATH = "data/bm25_data.pkl"

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25_engine = bm25_data["bm25"]
        bm25_docs = bm25_data["documents"]

    print("✅ 챗봇 엔진 로딩 완료!\n")

except Exception as e:
    print(f"⚠️ 챗봇 로딩 실패 (데이터 파일 경로를 확인하세요): {e}")
    sbert_model = None


# --- 도구 함수 ---
def simple_tokenizer(text):
    return text.split()

# --- 검색 함수들 ---
def search_vector(query, k=5):
    if not sbert_model: return []
    query_vec = sbert_model.encode(query).tolist()
    results = vector_collection.query(query_embeddings=[query_vec], n_results=k)
    output = []
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            output.append({
                "content": results['metadatas'][0][i].get("original_text", doc),
                "score": results['distances'][0][i], 
                "type": "vector"
            })
    return output

def search_bm25(query, k=5):
    if not sbert_model: return [] # 모델 로드 실패 시 중단
    tokenized_query = simple_tokenizer(query)
    scores = bm25_engine.get_scores(tokenized_query)
    top_n_indexes = np.argsort(scores)[::-1][:k] 
    output = []
    for idx in top_n_indexes:
        output.append({
            "content": bm25_docs[idx],
            "score": scores[idx],
            "type": "bm25"
        })
    return output

def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, item in enumerate(results):
            content = item['content']
            if content not in fused_scores:
                fused_scores[content] = 0
            fused_scores[content] += 1 / (rank + k)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in reranked_results]

# --- 최종 호출 함수 ---
def get_hybrid_answer(query: str):
    if not sbert_model:
        return "서버 오류: 챗봇 데이터가 로드되지 않았습니다."
        
    # 1. 벡터 + BM25 검색
    vec_results = search_vector(query, k=5)
    bm25_results = search_bm25(query, k=5)
    
    # 2. RRF 재정렬
    final_docs = reciprocal_rank_fusion([vec_results, bm25_results])[:3]
    
    # 3. 결과 종합
    context = "\n\n".join(final_docs)
    
    # (나중에 여기에 LLM 연결 코드를 넣으면 됩니다)
    return f"🤖 질문: {query}\n\n📚 [찾은 근거 자료]\n{context}"