# ==============================================================================
# SSU_25_NLP_project - chunked_db.py
#
# [개요]
# 'data.py'가 생성한 원본 SQLite DB의 긴 텍스트 문서들을 검색 효율을 위해
# 작은 단위의 '청크'로 분할하고 JSONL 파일로 출력하는 스크립트입니다.
#
# [주요 역할]
# 1. 문서 분할: LangChain의 RecursiveCharacterTextSplitter를 사용하여 문서를 쪼갬.
#    - 설정: CHUNK_SIZE=400, CHUNK_OVERLAP=50
# 2. 메타데이터 부여: 각 청크에 원본 출처(source), 제목, link 등의 메타데이터를 부여.
# 3. 중간 저장: 분할된 모든 청크를 'chunked_data.jsonl' 파일로 저장 (BM25/Vector DB의 입력).
#
# [처리 흐름]
# SQLite DB 로드 -> Text Splitter 적용 -> JSONL 파일 저장
# ==============================================================================
# 작업자 : 박채은(구조 분리 및 보완), 박원형(설계 및 구축)

import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DB_PATH = os.path.join(PROJECT_ROOT, "..", "ssu_chatbot_data.db")
CHUNKED_DATA_PATH = os.path.join(PROJECT_ROOT, "..", "chunked_data.jsonl")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def load_and_chunk_data(db_path):
    print(f"DB 연결 중: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    chunked_data = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

    # ----------------------
    # 1. 강의평 (청킹 X)
    # ----------------------
    cursor.execute("SELECT id, subject_name, professor_name, review_text, semester FROM lecture_reviews")
    reviews = cursor.fetchall()
    print(f" -> 'lecture_reviews' {len(reviews)}건 로딩")

    for row in reviews:
        text = f"과목명: {row[1]}, 교수명: {row[2]}, 강의평: {row[3]}"
        metadata = {
            "source": "lecture_review",
            "subject": row[1], "professor": row[2], "semester": row[4],
            "original_text": row[3],
            "new_id": f"review_{row[0]}"
        }
        chunked_data.append({"text": text, "metadata": metadata, "id":f"review_{row[0]}"})

    # ----------------------
    # 2. 공지사항 (청킹 O)
    # ----------------------
    cursor.execute("SELECT id, title, category, full_body_text, link, department FROM notices")
    notices = cursor.fetchall()
    print(f" -> 'notices' {len(notices)}건 로딩 및 청킹 중...")

    for row in notices:
        chunks = text_splitter.split_text(row[3])
        for i, chunk in enumerate(chunks):
            text = f"공지: {row[1]} (카테고리: {row[2]})\n내용: {chunk}"
            metadata = {
                "source": "notice",
                "title": row[1], "category": row[2], "department": row[5], "link": row[4],
                "original_text": chunk,
                "new_id": f"notice_{row[0]}_chunk_{i}"
            }
            chunked_data.append({"text": text, "metadata": metadata, "id":f"notice_{row[0]}_chunk_{i}"})

    # ----------------------
    # 3. 동아리 (청킹 O)
    # ----------------------
    try:
        cursor.execute("SELECT id, club_name, category, description, recruitment_info, source_url FROM clubs")
        clubs = cursor.fetchall()
        print(f" -> 'clubs' {len(clubs)}건 로딩 및 청킹 중...")

        for row in clubs:
            chunks = text_splitter.split_text(row[3])
            for i, chunk in enumerate(chunks):
                text = f"동아리: {row[1]} (분과: {row[2]})\n소개: {chunk}"
                metadata = {
                    "source": "club",
                    "club_name": row[1],
                    "category": row[2],
                    "link": row[5],
                    "original_text": chunk,
                    "new_id": f"club_{row[0]}_chunk_{i}"
                }
                chunked_data.append({"text": text, "metadata": metadata, "id": f"club_{row[0]}_chunk_{i}"})
    except Exception as e:
        print(f"[경고] 동아리 로드 중 오류: {e}")

    conn.close()

    # JSONL로 저장
    print(f"\n[저장] {len(chunked_data)}개의 청킹 결과를 '{CHUNKED_DATA_PATH}'에 저장 중...")
    with open(CHUNKED_DATA_PATH, 'w', encoding='utf-8') as f:
        for item in chunked_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("[OK] 청킹 완료.")

    return chunked_data


if __name__ == "__main__":
    load_and_chunk_data(SOURCE_DB_PATH)
