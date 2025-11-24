

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import sqlite3
import os
import unicodedata
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi
from openai import OpenAI
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# 프로젝트 루트 / DB / HF 캐시 / .env 설정
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


def fetch_soongguri_menu() -> str:
    """
    숭실대 생협(soongguri.com/m)의 현재 선택된 날짜 학식 메뉴를 파싱한다.
    - HTML 구조: <td class="menu_nm">중식1</td> + <td class="menu_list">안에 상세 구성
    """
    try:
        resp = requests.get(SOONGGURI_URL, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"[학식] soongguri 사이트 접속 실패: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    main_div = soup.find("div", id="mainDiv")
    if not main_div:
        return "[학식] soongguri 페이지에서 mainDiv를 찾지 못했습니다. HTML 구조를 다시 확인해주세요."

    menus: List[str] = []

    # 각 메뉴(중식1, 중식2, 석식1 등)는 한 줄(tr)에 menu_nm / menu_list 형식으로 들어있음
    for tr in main_div.find_all("tr"):
        name_td = tr.find("td", class_="menu_nm")
        list_td = tr.find("td", class_="menu_list")
        if not (name_td and list_td):
            continue

        meal_name = name_td.get_text(strip=True)  # 예: "중식1", "석식1"

        # 1) 코너 이름 (예: [뚝배기코너], [덮밥코너] 등)
        corner = ""
        first_block = list_td.find("div")
        if first_block:
            for tag in first_block.find_all(["font", "b", "span"], recursive=True):
                text = tag.get_text(strip=True)
                if "[" in text and "]" in text:
                    corner = text
                    break

        # 2) 메인 메뉴 이름 (예: ★ 차돌순두부찌개 - 5.0)
        main_dish = ""
        for tag in list_td.find_all(["font", "b", "span"], recursive=True):
            text = tag.get_text(" ", strip=True)
            if "★" in text:
                main_dish = text.replace("★", "").strip()
                break

        # 3) 반찬 / 구성 메뉴들 (작은 table의 td들)
        side_dishes: List[str] = []
        for td in list_td.find_all("td"):
            t = td.get_text(strip=True)
            if not t:
                continue
            if "알러지유발식품" in t or "원산지" in t:
                continue
            if t == "　":
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
        return "[학식] soongguri에서 오늘의 메뉴를 파싱하지 못했습니다."

    today_str = datetime.now().strftime("%Y-%m-%d")
    menu_text = f"[생협 식당 메뉴 - {today_str}]\n" + "\n\n".join(menus)
    return menu_text


def fetch_dorm_menu() -> str:
    """
    숭실대 기숙사 식당 주간 식단표(boxstyle02 테이블)에서 '오늘 날짜'에 해당하는
    중식/석식 메뉴를 파싱한다.
    - HTML 구조:
        <table class="boxstyle02">
          <tr> (헤더)
          <tr>
            <th> <a ...>2025-11-21 (금)</a> </th>
            <td>조식</td>
            <td>중식</td>
            <td>석식</td>
            <td>중.석식</td>
    """
    try:
        # 인증서 경고 회피용 verify=False (필요하면 True로 변경 가능)
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
        rows_text = []
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

try:
    KIWI_PROCESSOR = Kiwi()
except Exception as e:
    print(f"[ERROR] Kiwi 객체 초기화 실패: {e}")
    KIWI_PROCESSOR = None


def simple_tokenize(text: str) -> List[str]:
    """
    Kiwipiepy 형태소 분석기를 사용한 한국어 토크나이징.
    UnicodeDecodeError 방지를 위해 정규화 및 방어 코드 포함.
    """
    if not KIWI_PROCESSOR:
        return str(text or "").strip().split()

    if not text:
        return []

    text = str(text).strip()
    text = unicodedata.normalize("NFC", text)

    try:
        clean_text = text.encode("utf-8", "ignore").decode("utf-8")
    except Exception:
        clean_text = text

    if not clean_text:
        return []

    tokens: List[str] = []
    try:
        for token in KIWI_PROCESSOR.tokenize(clean_text, normalize_coda=True):
            if token.tag.startswith(("N", "V", "M", "SL", "SN")):
                tokens.append(token.form)
    except Exception as e:
        print(f"[Tokenize Warning] Skipped text due to error: {e}")
        return clean_text.split()

    return tokens


class BM25Retriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        print(f"[BM25] {len(docs)}개 문서 토큰화 시작...")
        self.corpus_tokens: List[List[str]] = [simple_tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"[BM25] 토큰화 및 인덱싱 완료.")

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

    def build_meal_prompt(self, query: str, meal_context: str) -> (str, str):
        """
        학식 전용 프롬프트.
        """
        system_msg = (
            "너는 숭실대학교 학식 정보를 알려주는 챗봇이야.\n"
            "아래에 제공된 학식 메뉴 정보만 근거로 한국어로 친절하게 답변해.\n"
            "모르겠으면 모른다고 말해."
        )

        user_msg = (
            f"사용자 질문:\n{query}\n\n"
            f"다음은 오늘 학식 관련 정보야. 이 정보만 근거로 답변을 만들어줘.\n\n"
            f"{meal_context}"
        )

        return system_msg, user_msg

    def answer_with_llm(
        self,
        query: str,
        llm_call: Callable[[str, str], str],
        intent: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
    ) -> str:
        # 1) 학식 intent면 RAG 대신 학식 컨텍스트 사용
        if intent == "학식_검색":
            meal_context = build_meal_context()
            system_msg, user_msg = self.build_meal_prompt(query, meal_context)
            answer = llm_call(system_msg, user_msg)
            return answer

        # 2) 그 외 intent는 RAG 흐름
        docs = self.retrieve(query, intent=intent, slots=slots)
        system_msg, user_msg = self.build_prompt(query, docs)
        answer = llm_call(system_msg, user_msg)

        # RAGAS 평가용 포맷이 필요하면 여기서 변환해서 반환하도록 수정 가능
        return answer


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
# 7. 간단 테스트용 main
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

        print("\n--- 🧠 LLM이 답변을 생성 중입니다... ---")

        # 매우 간단한 intent 예시 (실제 서비스에서는 NLU에서 넘겨줄 것)
        lower_q = q.lower()
        if "학식" in q or "메뉴" in q or "밥 뭐" in q:
            intent = "학식_검색"
        else:
            intent = None

        answer = rag.answer_with_llm(q, llm_call=call_openai_api, intent=intent)

        print("\n=======================================================")
        print(f"[궁금했슈(SSU) 답변]\n{answer}")
        print("=======================================================\n")