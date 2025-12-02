# ==============================================================================
# SSU_25_NLP_project - bm25_tokenize.py
#
# [개요]
# RAG 챗봇의 'BM25 검색' 모듈을 위해, 청크 데이터를 한국어 형태소 기반으로 토큰화하고
# BM25 전용 SQLite DB에 저장하는 인덱싱 스크립트입니다.
#
# [주요 역할]
# 1. 한국어 토큰화: Kiwi 형태소 분석기를 사용하여 텍스트에서 명사, 동사 등 핵심 품사만 추출.
# 2. 인덱스 구축: 토큰화된 결과를 'bm25_tokens.db'에 저장 (BM25Okapi가 사용할 코퍼스).
# 3. 성능 확보: 불용어(조사, 어미)를 제거하여 검색 인덱스의 크기를 줄이고 정확도를 높임.
#
# [처리 흐름]
# chunked_data.jsonl 로드 -> Kiwi 토큰화 -> bm25_tokens.db에 저장
# ==============================================================================
# 작업자 : 박채은

import os
import json
import sqlite3
import unicodedata
from kiwipiepy import Kiwi

# -------------------------------
# 경로 설정
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ssu/
JSONL_PATH = os.path.join(PROJECT_ROOT, "chunked_data.jsonl")
DB_PATH = os.path.join(PROJECT_ROOT, "bm25_tokens.db")

# -------------------------------
# Kiwi 초기화
# -------------------------------
try:
    KIWI = Kiwi()
except Exception as e:
    print(f"[ERROR] Kiwi 초기화 실패: {e}")
    KIWI = None

# -------------------------------
# 토큰화 함수
# -------------------------------
def tokenize(text: str):
    if not KIWI:
        return str(text or "").strip().split()
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)

    # UTF-8 안전하게
    try:
        clean_text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        clean_text = text

    tokens = []
    try:
        for token in KIWI.tokenize(clean_text, normalize_coda=True):
            if token.tag.startswith(("N", "V", "M", "SL", "SN")):
                tokens.append(token.form)
    except Exception:
        # Kiwi 실패 시 fallback
        tokens = clean_text.split()

    return tokens


# -------------------------------
# DB 초기화
# -------------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT,
    metadata TEXT,
    tokens TEXT
)
""")
conn.commit()

# -------------------------------
# JSONL → DB 저장
# -------------------------------
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        item = json.loads(line)
        _id = item["id"]
        text = item["text"]
        metadata = json.dumps(item.get("metadata", {}), ensure_ascii=False)
        tokens = " ".join(tokenize(text))

        cur.execute("""
        INSERT OR REPLACE INTO chunks (id, text, metadata, tokens)
        VALUES (?, ?, ?, ?)
        """, (_id, text, metadata, tokens))
        count += 1
        if count % 1000 == 0:
            print(f"{count} chunks 저장 완료...")

conn.commit()
conn.close()
print(f"[완료] 총 {count}개의 청크 DB 저장 완료: {DB_PATH}")




