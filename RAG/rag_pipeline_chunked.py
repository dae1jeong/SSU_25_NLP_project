# ==============================================================================
# SSU_25_NLP_project - RAG/rag_pipeline_chunked.py
#
# [개요]
# 이 파일은 RAG 챗봇의 최종 완성 버전으로, 프로젝트의 핵심 기술인
# '하이브리드 검색'을 구현한 메인 파이프라인 클래스입니다.
# main.py에서 이 클래스를 로드하여 서비스에 사용합니다.
#
# [주요 특징]
# 1. 하이브리드 검색: BM25 검색과 벡터 검색을 동시에 수행.
# 2. RRF 융합: 두 검색 결과를 RRF(Reciprocal Rank Fusion) 알고리즘으로 통합하여 최적의 문서 확보.
# 3. NLU 필터링: NLU가 파악한 의도(Intent) 및 개체명(Slot)을 사용해 검색 결과를 필터링.
# 4. 실시간 기능: 학식 관련 질문은 RAG를 건너뛰고 실시간 웹 스크래핑을 통해 즉시 응답.
#
# [사용 DB]
# - BM25DBRetriever: bm25_tokens.db (BM25 전용)
# - Vector Search: chroma_db (벡터 전용)
# ==============================================================================




from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import sqlite3
import os
from datetime import datetime
import json
import re  # 정규식
import time

import collections
import collections.abc
# bs4가 Python 3.12에서 collections.Callable을 참조해서 나는 오류 방지용 패치
if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable  # type: ignore[attr-defined]

import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

# 🔹 추가: Chroma + SBERT
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 환경 변수 (토크나이저 경고 줄이기용, 선택)
# ------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------
# 프로젝트 루트 / DB / HF 캐시 / .env 설정
# ------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "bm25_tokens.db")          # BM25 토큰 DB
CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_db")           # Chroma 벡터 DB
CHROMA_COLLECTION_NAME = "ssu_knowledge_base"
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"

HF_CACHE_DIR = os.path.join(ROOT_DIR, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR

print("[RAG] Using DB_PATH =", DB_PATH)
print("[RAG] Using HF cache dir =", HF_CACHE_DIR)
print("[RAG] Using CHROMA_PATH =", CHROMA_PATH)

env_path = os.path.join(ROOT_DIR, ".env")
print(f"[RAG] Loading .env from: {env_path}")

if os.path.exists(env_path):
    load_dotenv(env_path)
    print("[RAG] .env 파일 로드 성공")
else:
    print("[RAG] 🚨 경고: .env 파일을 찾을 수 없습니다!")

# ==========================================================
# 0. 학식 관련 스크래핑 유틸 (soongguri / 기숙사 식당)
# ==========================================================

SOONGGURI_URL = "https://soongguri.com/m/"
DORM_FOOD_URL = (
    "https://ssudorm.ssu.ac.kr:444/"
    "SShostel/mall_main.php?viewform=B0001_foodboard_list&board_no=1"
)

# soongguri가 모바일 브라우저라고 믿도록 헤더 세팅
SOONGGURI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _clean_line(text: str) -> str:
    """공백 정리 + 쓸모없는 기호 제거용 유틸."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_sdt(date_str: str | None) -> tuple[str, str]:
    """
    date_str:
      - None        → 오늘 날짜
      - '20251125'  → 그대로 사용
      - '2025-11-25' → '-' 제거 후 사용
    return: (sdt, pretty_label)
    """
    if not date_str:
        dt = datetime.now()
        sdt = dt.strftime("%Y%m%d")
        label = dt.strftime("%Y-%m-%d")
        return sdt, label

    ds = date_str.strip()

    # 2025-11-25 형식
    if len(ds) == 10 and ds[4] == "-" and ds[7] == "-":
        try:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 20251125 형식
    if len(ds) == 8 and ds.isdigit():
        try:
            dt = datetime.strptime(ds, "%Y%m%d")
            return ds, dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 이상하면 오늘 날짜로 fallback
    dt = datetime.now()
    return dt.strftime("%Y%m%d"), dt.strftime("%Y-%m-%d")


def fetch_soongguri_menu(date_str: str | None = None, rcd: str = "1") -> str:
    """
    soongguri AJAX 엔드포인트(/m/m_req/m_menu.php)에서
    주어진 날짜(date_str)와 식당 코드(rcd)의 메뉴 HTML을 직접 가져와서 파싱.
    """
    sdt, label = _normalize_sdt(date_str)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        session = requests.Session()
        session.headers.update(SOONGGURI_HEADERS)

        # 실제 AJAX 메뉴 데이터 요청 (m_menu.php)
        params = {"rcd": rcd, "sdt": sdt}
        resp = session.get(
            SOONGGURI_URL + "m_req/m_menu.php",
            params=params,
            timeout=10,
            verify=False,
        )
    except Exception as e:
        return (
            "[생협 식당 메뉴]\n"
            "soongguri 사이트에 접속하지 못했어요.\n"
            f"(에러: {e})\n"
            "→ 직접 확인: https://soongguri.com/m/"
        )

    if resp.status_code != 200 or len(resp.text.strip()) < 50:
        return (
            f"[생협 식당 메뉴 - {label}]\n"
            "현재 soongguri에서 학식 정보를 가져오지 못했어요.\n"
            "→ 직접 확인: https://soongguri.com/m/"
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")

    if not table:
        return (
            f"[생협 식당 메뉴 - {label}]\n"
            "식단 테이블을 찾지 못했어요.\n"
            "→ 직접 확인: https://soongguri.com/m/"
        )

    menus: list[str] = []

    for tr in table.find_all("tr"):
        name_td = tr.find("td", class_="menu_nm")
        list_td = tr.find("td", class_="menu_list")
        if not (name_td and list_td):
            continue

        meal_name = name_td.get_text(strip=True)  # 예: 중식1, 석식1...

        # 코너명 [뚝배기코너] 등
        corner = ""
        for tag in list_td.find_all(["font", "b", "span"]):
            txt = tag.get_text(strip=True)
            m = re.search(r"\[[^\]]+\]", txt)
            if m:
                corner = m.group(0)
                break

        # 메인 메뉴 (★ 표시)
        main_dish = ""
        for tag in list_td.find_all(["font", "b", "span"]):
            txt = tag.get_text(" ", strip=True)
            if "★" in txt:
                main_dish = txt.replace("★", "").strip()
                break

        # 반찬 후보들
        side_dishes: list[str] = []
        for li in list_td.select("ul.mean_list li, ul.mean_list td, ul.mean_list .xl65"):
            t = _clean_line(li.get_text(strip=True))
            if not t:
                continue
            if "알러지유발식품" in t or "원산지" in t:
                continue
            if all(ord(ch) < 128 for ch in t):  # 전부 ASCII(영문)면 스킵
                continue
            if t not in side_dishes:
                side_dishes.append(t)

        line = meal_name
        if corner:
            line += f" {corner}"
        if main_dish:
            line += f" - {main_dish}"
        if side_dishes:
            line += "\n  · " + "\n  · ".join(side_dishes)

        menus.append(line)

    if not menus:
        return (
            f"[생협 식당 메뉴 - {label}]\n"
            "메뉴 파싱 실패 (항목 없음)\n"
            "→ https://soongguri.com/m/ 에서 직접 확인해 주세요."
        )

    return f"[생협 식당 메뉴 - {label}]\n" + "\n\n".join(menus)


def fetch_dorm_menu() -> str:
    """
    숭실대 기숙사 식당 주간 식단표(boxstyle02 테이블)에서 '오늘 날짜'에 해당하는
    조식/중식/석식 메뉴를 파싱한다.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        resp = requests.get(DORM_FOOD_URL, timeout=10, verify=False)
        resp.raise_for_status()
    except Exception as e:
        return f"[학식] 기숙사 식당 사이트 접속 실패: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="boxstyle02")
    if not table:
        return "[학식] 기숙사 식당 메뉴 테이블(boxstyle02)을 찾지 못했습니다."

    today = datetime.now().strftime("%Y-%m-%d")
    target_row = None

    # 첫 번째 tr는 헤더, 그 다음부터 실제 날짜 행
    for tr in table.find_all("tr")[1:]:
        th = tr.find("th")
        if not th:
            continue
        text = th.get_text(strip=True)  # 예: "2025-11-21 (금)"
        if today in text:
            target_row = tr
            break

    # 오늘 날짜가 없으면, 주간 메뉴 요약 반환
    if target_row is None:
        rows_text: list[str] = []
        for tr in table.find_all("tr")[1:]:
            th = tr.find("th")
            if not th:
                continue
            date_text = th.get_text(strip=True)
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            breakfast = tds[0].get_text("\n", strip=True)
            lunch = tds[1].get_text("\n", strip=True)
            dinner = tds[2].get_text("\n", strip=True)
            row_str = (
                f"{date_text}\n"
                f"  · 조식: {breakfast}\n"
                f"  · 중식: {lunch}\n"
                f"  · 석식: {dinner}"
            )
            rows_text.append(row_str)

        if not rows_text:
            return "[학식] 기숙사 식당 주간 메뉴를 파싱하지 못했습니다."
        return "[기숙사 식당 주간 메뉴]\n\n" + "\n\n".join(rows_text)

    # 오늘 날짜 행을 찾은 경우
    tds = target_row.find_all("td")
    breakfast = tds[0].get_text("\n", strip=True) if len(tds) >= 1 else ""
    lunch = tds[1].get_text("\n", strip=True) if len(tds) >= 2 else ""
    dinner = tds[2].get_text("\n", strip=True) if len(tds) >= 3 else ""
    both = tds[3].get_text("\n", strip=True) if len(tds) >= 4 else ""

    date_label = target_row.find("th").get_text(strip=True)

    lines = [
        f"[기숙사 식당 메뉴 - {date_label}]",
        f"· 조식: {breakfast or '미운영'}",
        f"· 중식: {lunch or '미등록'}",
        f"· 석식: {dinner or '미등록'}",
    ]
    if both:
        lines.append(f"· 중·석식: {both}")

    return "\n".join(lines)


def build_meal_context() -> str:
    """
    soongguri(생협) + 기숙사 식당 메뉴를 하나의 텍스트로 합침.
    intent == "학식_검색"일 때 이 문자열을 LLM에 넘긴다.
    """
    soongguri_text = fetch_soongguri_menu()
    dorm_text = fetch_dorm_menu()

    context_parts = [
        "다음은 숭실대학교 오늘의 학식 관련 정보입니다.",
        "",
        soongguri_text,
        "",
        dorm_text,
    ]
    return "\n".join(context_parts)

# ======================== Eval을 위한 코드 ====================
# ragas, Recall@K 등 다른 평가를 위한 리턴 타입 정의
@dataclass
class EvaluationResult:
    """RAG 시스템 평가에 필요한 모든 결과를 담는 구조"""
    query: str                       
    model_answer: str                
    retrieved_chunks: List['ChunkDocument'] 
    context_texts: List[str]         
    is_rag_flow: bool
    latency_seconds: float = 0.0 


# ==============================================
# 1. BM25 / RAG 파트
# ==============================================

@dataclass
class ChunkDocument:
    id: str
    text: str
    meta: Dict
    tokens: List[str]


def load_chunks_from_db(db_path: str = DB_PATH) -> List[ChunkDocument]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, text, metadata, tokens FROM chunks")
    rows = cur.fetchall()
    conn.close()

    docs: List[ChunkDocument] = []
    for id_, text, metadata_json, tokens_str in rows:
        meta = json.loads(metadata_json)
        tokens = tokens_str.split()
        docs.append(ChunkDocument(id=id_, text=text, meta=meta, tokens=tokens))
    print(f"[DB] 총 {len(docs)}개의 청크 로드 완료")
    return docs


class BM25DBRetriever:
    def __init__(self, chunk_docs: List[ChunkDocument]):
        self.docs = chunk_docs
        self.corpus_tokens = [d.tokens for d in chunk_docs]  # DB에 있는 토큰 사용
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"[BM25] DB 토큰으로 BM25 인덱스 생성 완료")

    def search(self, query: str, top_k: int = 30) -> List[ChunkDocument]:
        query_tokens = query.strip().split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(-scores)[:top_k]
        return [self.docs[i] for i in ranked_indices]


# (기존 VectorReranker는 BM25 상위 k만 자르는 용도라서, 하이브리드 검색 도입 후 사용하지 않음.
# 필요하면 남겨두고, 실제 호출은 하지 않는다.)
class VectorReranker:
    """
    ⚠️ (현재 미사용)
    세그폴트 방지를 위해 sentence_transformers 모델을 사용하지 않고
    BM25 결과를 그대로 상위 top_k만 잘라서 반환하는 단순한 reranker.
    """

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        print("[VectorReranker] (미사용) BM25 순서 그대로 사용합니다.")
        self.model = None  # 실제 모델 로딩 안 함

    def rerank(self, query: str, candidates: List[ChunkDocument], top_k: int = 5) -> List[ChunkDocument]:
        return candidates[:top_k]


class RAGPipeline:
    def __init__(self, bm25_top_k: int = 30, rerank_top_k: int = 5):
        # 1) BM25용 청크 로드
        self.chunk_docs = load_chunks_from_db()
        self.bm25 = BM25DBRetriever(self.chunk_docs)
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k

        # id → ChunkDocument 매핑 (RRF에서 사용)
        self.id_to_doc: Dict[str, ChunkDocument] = {d.id: d for d in self.chunk_docs}

        # 2) ChromaDB 연결 (벡터 검색용)
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME
            )
            print("[Chroma] 컬렉션 로드 완료:", CHROMA_COLLECTION_NAME)
        except Exception as e:
            print(f"[Chroma] 로드 실패: {e}")
            self.chroma_client = None
            self.chroma_collection = None

        # 3) 쿼리 임베딩용 SBERT 
        try:
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[Embedding] SentenceTransformer 로드 완료: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"[Embedding] SentenceTransformer 로드 실패: {e}")
            self.encoder = None

    # -------------------------
    # 1) 벡터 검색 (Chroma)
    # -------------------------
    def vector_search(self, query: str, top_k: int = 30) -> List[ChunkDocument]:
        """
        ChromaDB에 저장된 임베딩 기반 벡터 검색
        """
        if self.chroma_collection is None or self.encoder is None:
            return []

        try:
            query_emb = self.encoder.encode([query])  # (1, embedding_dim)
            res = self.chroma_collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=top_k,
                include=["metadatas"],   # 🔥 ids는 최신버전에서 제거됨 
            )
        except Exception as e:
            print(f"[Chroma] vector_search 실패: {e}")
            return []

        metas = res.get("metadatas", [[]])[0]

        ids = []
        for m in metas:
            cid = m.get("new_id")  # vector_db.py에서 저장한 실제 chunk ID
            if cid:
                ids.append(cid)

        docs: List[ChunkDocument] = []
        for cid in ids:
            doc = self.id_to_doc.get(cid)
            if doc:
                docs.append(doc)

        return docs

    # -------------------------
    # 2) RRF 결합 함수
    # -------------------------
    def _rrf_merge(
        self,
        bm25_docs: List[ChunkDocument],
        vec_docs: List[ChunkDocument],
        top_k: int = 30,
        k: int = 60,
    ) -> List[ChunkDocument]:
        """
        Reciprocal Rank Fusion:
        각 랭크에 대해 1 / (k + rank)를 더해 점수를 합산한 뒤,
        최종 점수 순으로 정렬해서 상위 top_k를 반환한다.
        """
        scores: Dict[str, float] = {}

        # BM25 순위
        for rank, d in enumerate(bm25_docs):
            scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (k + rank + 1)

        # 벡터 검색 순위
        for rank, d in enumerate(vec_docs):
            scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (k + rank + 1)

        # 점수 기준으로 정렬
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        merged_docs: List[ChunkDocument] = []
        for cid, _score in sorted_ids:
            doc = self.id_to_doc.get(cid)
            if doc and doc not in merged_docs:
                merged_docs.append(doc)
            if len(merged_docs) >= top_k:
                break

        return merged_docs

    # -------------------------
    # 3) intent/slot 필터
    # -------------------------
    def _apply_filters(
        self,
        candidates: List[ChunkDocument],
        intent: Optional[str],
        slots: Dict,
    ) -> List[ChunkDocument]:
        if intent:
            candidates = [
                d for d in candidates
                if (intent == "강의평_검색" and d.meta.get("source") == "lecture_review")
                or (intent == "공지_검색" and d.meta.get("source") == "notice")
                or (intent == "동아리_검색" and d.meta.get("source") == "club")
            ]

        prof = slots.get("professor_name") or slots.get("professor")
        if prof:
            candidates = [
                d for d in candidates
                if "professor" in d.meta and prof in d.meta["professor"]
            ]

        dept = slots.get("department")
        if dept:
            candidates = [
                d for d in candidates
                if "department" in d.meta and dept in d.meta["department"]
            ]

        club_name = slots.get("club_name")
        if club_name:
            candidates = [
                d for d in candidates
                if "club_name" in d.meta and club_name in d.meta["club_name"]
            ]

        return candidates

    # -------------------------
    # 4) 최종 Retrieve (BM25 + 벡터 + RRF)
    # -------------------------
    def retrieve(self, query: str, intent: str = None, slots: Dict = None):
        slots = slots or {}

        # 1) BM25 후보 검색
        bm25_candidates = self.bm25.search(query, top_k=self.bm25_top_k)

        # 2) 벡터 검색 후보
        vector_candidates = self.vector_search(query, top_k=self.bm25_top_k)

        # 3) RRF로 두 결과 결합 (벡터 검색이 실패하면 BM25만 사용)
        if vector_candidates:
            merged = self._rrf_merge(
                bm25_candidates,
                vector_candidates,
                top_k=self.bm25_top_k,
            )
        else:
            merged = bm25_candidates

        # 4) intent / slot 필터
        candidates = self._apply_filters(merged, intent=intent, slots=slots)

        # 5) 필터 결과가 비면 BM25 원본으로 fallback
        if not candidates:
            candidates = bm25_candidates

        # 6) 최종 top_k 반환
        return candidates[:self.rerank_top_k]

    # -------------------------
    # 5) 프롬프트 구성
    # -------------------------
    def build_prompt(self, query: str, docs: List[ChunkDocument]) -> tuple[str, str]:
        context_blocks = []
        for i, d in enumerate(docs, start=1):
            header = f"[문서 {i} | {d.meta.get('source', 'unknown')} | id={d.id}]"
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

    # -------------------------
    # 6) LLM 호출 래퍼
    # -------------------------
    def answer_with_llm(
        self,
        query: str,
        llm_call: Callable[[str, str], str],
        intent: str = None,
        slots: Dict = None,
    ) -> str:
        """
        질문/의도에 따라 학식 스크래핑 또는 일반 RAG를 사용하여 답변 생성.
        """

        # ✅ 1) 질문 문자열만 보고 '학식' 관련 의도 자동 판별
        if (
            ("학식" in query)
            or ("메뉴" in query)
            or ("밥 뭐" in query)
            or ("밥 뭐 나와" in query)
            or ("오늘 밥" in query)
            or ("생협" in query)
            or ("기숙사 식당" in query)
        ):
            intent = "학식_검색"

        # ✅ 2) 학식 의도면 RAG 말고 실시간 스크래핑 컨텍스트 사용
        if intent == "학식_검색":
            meal_context = build_meal_context()
            system_msg = (
                "너는 숭실대학교 학식 정보를 알려주는 챗봇이야.\n"
                "아래 컨텍스트(생협/기숙사 식당 메뉴)를 참고해서, "
                "사용자 질문에 맞게 오늘의 학식 정보를 간략하고 보기 좋게 정리해서 알려줘.\n"
                "메뉴 이름, 코너 이름, 가격, 끼니(조식/중식/석식) 등을 정돈해서 한국어로 친절하게 설명해."
            )
            user_msg = (
                f"사용자 질문: {query}\n\n"
                f"다음은 오늘의 학식 정보야:\n\n{meal_context}"
            )
            return llm_call(system_msg, user_msg)

        # ✅ 3) 그 외는 기존 RAG 파이프라인 사용
        docs = self.retrieve(query, intent=intent, slots=slots or {})
        system_msg, user_msg = self.build_prompt(query, docs)
        return llm_call(system_msg, user_msg)

# ======================== Eval을 위한 코드 ====================
# ragas, Recall@K 등 다른 평가를 위한 리턴 타입 정의
def answer_with_llm_EVAL(
        self,
        query: str,
        llm_call: Callable[[str, str], str],
        intent: str = None,
        slots: Dict = None,
    ) -> EvaluationResult: # 💡 리턴 타입: EvaluationResult
        """
        평가 작업을 위해 답변 외에 검색된 청크 정보를 포함하여 반환합니다.
        (기존 answer_with_llm의 로직을 그대로 사용하되, 리턴만 변경)
        """
        slots = slots or {}

        # ✅ 1) 질문 문자열만 보고 '학식' 관련 의도 자동 판별
        if (
            ("학식" in query)
            or ("메뉴" in query)
            or ("밥 뭐" in query)
            or ("밥 뭐 나와" in query)
            or ("오늘 밥" in query)
            or ("생협" in query)
            or ("기숙사 식당" in query)
        ):
            intent = "학식_검색"

        # ✅ 2) 학식 의도면 RAG 말고 실시간 스크래핑 컨텍스트 사용
        if intent == "학식_검색":
            meal_context = build_meal_context()
            system_msg = (
                "너는 숭실대학교 학식 정보를 알려주는 챗봇이야.\n"
                "아래 컨텍스트(생협/기숙사 식당 메뉴)를 참고해서, "
                "사용자 질문에 맞게 오늘의 학식 정보를 간략하고 보기 좋게 정리해서 알려줘.\n"
                "메뉴 이름, 코너 이름, 가격, 끼니(조식/중식/석식) 등을 정돈해서 한국어로 친절하게 설명해."
            )
            user_msg = (
                f"사용자 질문: {query}\n\n"
                f"다음은 오늘의 학식 정보야:\n\n{meal_context}"
            )
            final_answer = llm_call(system_msg, user_msg)
            
            # 💡 EvaluationResult 반환 (RAG flow 아님)
            return EvaluationResult(
                query=query,
                model_answer=final_answer,
                retrieved_chunks=[],
                context_texts=[],
                is_rag_flow=False
            )

        # ✅ 3) 그 외는 기존 RAG 파이프라인 사용
        
        # 1. 검색 결과 확보 (docs 변수에 저장)
        docs: List[ChunkDocument] = self.retrieve(query, intent=intent, slots=slots)
        
        # 2. LLM 호출을 위한 프롬프트 구성
        system_msg, user_msg = self.build_prompt(query, docs)
        
        # 3. LLM 답변 생성
        start_time = time.time()  # 💡 측정 시작
        final_answer = llm_call(system_msg, user_msg)
        end_time = time.time()    # 💡 측정 종료

        latency = end_time - start_time # 💡 소요 시간 (초 단위)

        # 4. RAGAs 'contexts'를 위해 텍스트 목록 추출
        context_texts = [d.text for d in docs]
        
        # 💡 최종적으로 EvaluationResult 객체를 반환
        return EvaluationResult(
            query=query,
            model_answer=final_answer,
            retrieved_chunks=docs,
            context_texts=context_texts,
            is_rag_flow=True,
            latency_seconds=latency # 💡 새로운 필드 추가
        )



# =========================
# 6. GPT API 호출 함수
# =========================

def call_openai_api(system_msg: str, user_msg: str) -> str:
    """
    OpenAI API를 호출하여 최종 답변을 생성합니다.
    .env 파일에서 키를 로드하므로 보안상 안전합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "[오류] .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다. .env 파일을 확인해주세요."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM 호출 오류] API 호출 중 오류 발생: {e}"





# =========================
# 8. 간단 CLI 테스트용 main
# =========================

if __name__ == "__main__":
    print(f"[RAG] Using DB_PATH = {DB_PATH}")
    print("터미널에서 직접 테스트합니다. '학식'이라고 쳐보세요.\n")

    while True:
        try:
            q = input("\n질문을 입력하세요 (종료: 엔터만 입력): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        print("\n--- 🧠 LLM이 답변을 생성 중입니다... ---")

        # 여기서 intent는 굳이 안 줘도 되지만, 넣어도 상관 없음
        if ("학식" in q) or ("메뉴" in q) or ("밥 뭐" in q):
            intent = "학식_검색"
        else:
            intent = None

        answer = rag.answer_with_llm(q, llm_call=call_openai_api, intent=intent)

        print("\n=======================================================")
        print(f"[궁금했슈(SSU) 답변]\n{answer}")
        print("=======================================================\n")