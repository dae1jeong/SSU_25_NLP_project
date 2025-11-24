# save_chunked_to_db.py
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




