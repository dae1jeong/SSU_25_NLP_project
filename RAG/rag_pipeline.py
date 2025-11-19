"""
RAG 파이프라인 예시 (SSU_25_NLP_PROJECT용 수정본)

역할:
    - 프로젝트 루트의 ssu_chatbot_data.db 에서 공지/강의평/동아리 데이터를 불러와 Document로 변환
    - BM25로 1차 검색, sentence-transformers로 2차 벡터 re-rank
    - LLM에 줄 컨텍스트 프롬프트 생성

주의:
    - 이 파일은 RAG 폴더 안에 있다고 가정합니다: SSU_25_NLP_PROJECT/RAG/rag_pipeline.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import sqlite3
import math
import os

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# 🌟 Kiwipiepy import 추가
from kiwipiepy import Kiwi

# ------------------------------------------------------------------
# !! 여기 중요 !!
# RAG 폴더 기준으로 한 칸 올라가서 루트의 DB를 바라보게 함
# ------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "ssu_chatbot_data.db")

HF_CACHE_DIR = os.path.join(ROOT_DIR, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR

print("[RAG] Using DB_PATH =", DB_PATH)
print("[RAG] Using HF cache dir =", HF_CACHE_DIR)

# =========================
# 1. Document 스키마 정의
# =========================

@dataclass
class Document:
    id: str               # "notice:123", "review:45" 같은 내부 ID
    type: str             # "notice" | "review" | "club"
    text: str             # 검색과 LLM 컨텍스트에 사용할 본문
    meta: Dict[str, str]  # 부가 정보 (학과, 교수명, 날짜, 동아리 이름 등)


# =========================
# 2. DB → Document 로더
# =========================

def load_notices() -> List[Document]:
    """
    notices 테이블에서 공지 데이터를 읽어와 Document 리스트로 변환.
    스키마는 data/data.py 기준:

        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        category TEXT,
        post_date DATE,
        status TEXT,
        full_body_text TEXT,
        link TEXT UNIQUE,
        department TEXT,
        created_at TIMESTAMP ...
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, category, post_date, status, full_body_text, link, department
        FROM notices
    """)
    rows = cur.fetchall()
    conn.close()

    docs: List[Document] = []
    for id_, title, category, post_date, status, body, link, dept in rows:
        title = title or ""
        category = category or ""
        post_date = post_date or ""
        status = status or ""
        body = body or ""
        dept = dept or "정보 없음"
        link = link or ""

        text = (
            f"[공지] {title}\n"
            f"- 카테고리: {category}\n"
            f"- 학과: {dept}\n"
            f"- 게시일: {post_date}\n"
            f"- 상태: {status}\n\n"
            f"{body}"
        )
        meta = {
            "title": title,
            "category": category,
            "post_date": post_date,
            "status": status,
            "department": dept,
            "link": link,
        }
        docs.append(
            Document(
                id=f"notice:{id_}",
                type="notice",
                text=text,
                meta=meta,
            )
        )
    return docs


def load_reviews() -> List[Document]:
    """
    lecture_reviews 테이블에서 강의평 데이터를 읽어와 Document 리스트로 변환.
    스키마:

        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_name TEXT,
        professor_name TEXT,
        star_rating REAL,
        semester TEXT,
        review_text TEXT UNIQUE,
        created_at TIMESTAMP ...
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, subject_name, professor_name, star_rating, semester, review_text
        FROM lecture_reviews
    """)
    rows = cur.fetchall()
    conn.close()

    docs: List[Document] = []
    for id_, subj, prof, star, sem, review_text in rows:
        subj = subj or ""
        prof = prof or "정보 없음"
        sem = sem or ""
        review_text = review_text or ""
        star = star if star is not None else 0.0

        text = (
            f"[강의평] {subj} - {prof} 교수님\n"
            f"- 별점: {star}\n"
            f"- 수강 학기: {sem}\n\n"
            f"{review_text}"
        )
        meta = {
            "subject_name": subj,
            "professor_name": prof,
            "star_rating": str(star),
            "semester": sem,
        }
        docs.append(
            Document(
                id=f"review:{id_}",
                type="review",
                text=text,
                meta=meta,
            )
        )
    return docs


def load_clubs() -> List[Document]:
    """
    clubs 테이블에서 동아리 데이터를 읽어와 Document 리스트로 변환.
    스키마:

        id INTEGER PRIMARY KEY AUTOINCREMENT,
        club_name TEXT,
        category TEXT,
        description TEXT,
        recruitment_info TEXT,
        source_url TEXT UNIQUE,
        created_at TIMESTAMP ...
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, club_name, category, description, recruitment_info, source_url
        FROM clubs
    """)
    rows = cur.fetchall()
    conn.close()

    docs: List[Document] = []
    for id_, name, category, desc, recruit, url in rows:
        name = name or "제목 없음"
        category = category or "동아리"
        desc = desc or ""
        recruit = recruit or ""
        url = url or ""

        text = (
            f"[동아리] {name} (분류: {category})\n"
            f"- 모집 정보: {recruit}\n"
            f"- 링크: {url}\n\n"
            f"{desc}"
        )
        meta = {
            "club_name": name,
            "category": category,
            "recruitment_info": recruit,
            "source_url": url,
        }
        docs.append(
            Document(
                id=f"club:{id_}",
                type="club",
                text=text,
                meta=meta,
            )
        )
    return docs


def load_all_docs() -> List[Document]:
    """
    공지 + 강의평 + 동아리 Document를 한 번에 로딩.
    """
    notices = load_notices()
    reviews = load_reviews()
    clubs = load_clubs()
    print(
        f"[RAG] loaded notices={len(notices)}, "
        f"reviews={len(reviews)}, clubs={len(clubs)}"
    )
    return notices + reviews + clubs


# =========================
# 3. BM25 1차 검색기
# =========================

# 🌟 Kiwi 객체를 전역 또는 클래스 레벨에서 초기화 (단 한 번만 로딩)

try:
    KIWI_PROCESSOR = Kiwi()
except Exception as e:
    print(f"[ERROR] Kiwi 객체 초기화 실패: {e}")
    KIWI_PROCESSOR = None # 실패 시 fallback 처리

def simple_tokenize(text: str) -> List[str]:
    """
    Kiwipiepy 형태소 분석기를 사용하여 한국어에 특화된 토크나이징.
    **UnicodeDecodeError 방지를 위한 전처리 추가.**
    """
    if not KIWI_PROCESSOR:
        return text.strip().split()

    # 🌟🌟🌟 오류 방지를 위한 핵심 전처리 🌟🌟🌟
    # 1. 텍스트가 None이 아닌지 확인하고 str로 변환
    text = str(text or "").strip()
    
    # 2. 유니코드 오류가 있는 경우 강제로 무시하고 클린 텍스트 생성
    try:
        # 대부분의 한국어 데이터는 'utf-8'이므로, 인코딩/디코딩 과정을 거쳐 오류 문자 제거
        clean_text = text.encode('utf-8', 'ignore').decode('utf-8')
    except Exception:
        # 혹시 모를 예외 발생 시 원본 텍스트 사용
        clean_text = text

    if not clean_text:
        return []
    # 🌟🌟🌟 전처리 종료 🌟🌟🌟
    

    tokens: List[str] = []
    
    # 🌟 clean_text 사용
    for token in KIWI_PROCESSOR.tokenize(clean_text, normalize_coda=True):
        if token.tag.startswith(('N', 'V', 'M', 'SL', 'SN')):
            tokens.append(token.form)
            
    return tokens

class BM25Retriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.corpus_tokens: List[List[str]] = [simple_tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 30) -> List[Document]:
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(self.docs)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        return [self.docs[i] for i in ranked_indices]


# =========================
# 4. 벡터 기반 재정렬기
# =========================

class VectorReranker:
    """
    SentenceTransformer로 후보 문서들을 벡터화하고
    cosine similarity로 재정렬.
    """
    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
    ):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _cosine_sim(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
        sims = d @ q
        return sims

    def rerank(self, query: str, candidates: List[Document], top_k: int = 5) -> List[Document]:
        if not candidates:
            return []

        texts = [d.text for d in candidates]
        doc_embs = self.model.encode(texts, convert_to_numpy=True)
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        sims = self._cosine_sim(query_emb, doc_embs)

        ranked_indices = np.argsort(-sims)[:top_k]
        return [candidates[i] for i in ranked_indices]


# =========================
# 5. RAG 파이프라인 클래스
# =========================

class RAGPipeline:
    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        bm25_top_k: int = 30,
        rerank_top_k: int = 5,
    ):
        self.docs = docs or load_all_docs()
        self.bm25 = BM25Retriever(self.docs)
        self.reranker = VectorReranker()
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        intent: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        slots = slots or {}

        # 1) BM25 후보
        candidates = self.bm25.search(query, top_k=self.bm25_top_k)

        # 2) intent/slots 기반 필터
        filtered: List[Document] = candidates

        if intent == "강의평_검색":
            filtered = [d for d in filtered if d.type == "review"]
        elif intent == "공지_검색":
            filtered = [d for d in filtered if d.type == "notice"]
        elif intent == "동아리_검색":
            filtered = [d for d in filtered if d.type == "club"]

        prof = slots.get("professor_name") or slots.get("professor")
        if prof:
            filtered = [
                d for d in filtered
                if d.meta.get("professor_name") and prof in d.meta.get("professor_name", "")
            ]

        dept = slots.get("department")
        if dept:
            filtered = [
                d for d in filtered
                if d.meta.get("department") and dept in d.meta.get("department", "")
            ]

        club_name = slots.get("club_name")
        if club_name:
            filtered = [
                d for d in filtered
                if d.meta.get("club_name") and club_name in d.meta.get("club_name", "")
            ]

        if not filtered:
            filtered = candidates

        # 3) 벡터 기반 재정렬
        final_docs = self.reranker.rerank(query, filtered, top_k=self.rerank_top_k)
        return final_docs

    def build_prompt(self, query: str, docs: List[Document]) -> (str, str):
        context_blocks = []
        for i, d in enumerate(docs, start=1):
            header = f"[문서 {i} | {d.type} | id={d.id}]"
            block = f"{header}\n{d.text}"
            context_blocks.append(block)

        context_text = "\n\n---\n\n".join(context_blocks)

        system_msg = (
            "너는 숭실대학교 관련 정보만 답변하는 챗봇이야.\n"
            "아래에 제공된 컨텍스트 안에서만 근거를 찾아서 한국어로 친절하게 답변해.\n"
            "모르겠으면 모른다고 말해."
        )

        user_msg = (
            f"사용자 질문:\n{query}\n\n"
            f"다음은 관련 문서들이야. 이 정보만 근거로 답변을 만들어줘.\n\n"
            f"{context_text}"
        )

        return system_msg, user_msg

    def answer_with_llm(
        self,
        query: str,
        llm_call: Callable[[str, str], str],
        intent: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
    ) -> str:
        docs = self.retrieve(query, intent=intent, slots=slots)
        system_msg, user_msg = self.build_prompt(query, docs)
        answer = llm_call(system_msg, user_msg)
        return answer


# =========================
# 6. 간단 테스트용 main
# =========================

if __name__ == "__main__":
    print(f"[RAG] Using DB_PATH = {DB_PATH}")
    rag = RAGPipeline()

    while True:
        try:
            q = input("\n질문을 입력하세요 (종료: 엔터만 입력): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        docs = rag.retrieve(q)
        print(f"\n[검색된 문서 수: {len(docs)}]\n")
        for i, d in enumerate(docs, start=1):
            print("=" * 80)
            print(f"[문서 {i}] id={d.id}, type={d.type}")
            print(d.text[:500])
            if len(d.text) > 500:
                print("... (생략)")
        print("\n--- 여기까지가 RAG 컨텍스트입니다. ---\n")