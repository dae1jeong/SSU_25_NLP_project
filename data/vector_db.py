#data.py를 통해 크롤링한 데이터를 1차적으로 전처리하고 처리한 데이터 베이스에서 데이트를 로드
#SentenceTransformer 모델을 이용해 모든 텍스트를 벡터로 변환
#변환된 벡터와 원본 테스트의 메타데이터를 벡터 db에 저장





# ==============================================================================
# SSU_25_NLP_project - data/vector_db.py
#
# [역할]
# RAG 챗봇의 '검색 엔진(Retriever)' 역할을 하는 벡터 데이터베이스를 구축하는 스크립트입니다.
# 'data.py'가 생성한 원본 DB(SQLite)에서 텍스트를 읽어와 임베딩(벡터화) 후 저장합니다.
#
# [처리 흐름]
# 1. (Load) SQLite DB에서 3가지 데이터(강의평, 공지사항, 동아리)를 로드합니다.
# 2. (Chunking) 긴 텍스트(공지사항, 동아리)는 검색 정확도를 위해 작은 단위로 쪼갭니다.
# 3. (Embedding) SBERT 모델을 사용해 텍스트를 768차원 숫자 벡터로 변환합니다.
# 4. (Save) 벡터와 메타데이터를 ChromaDB에 저장합니다. (대용량 배치 처리 포함)
# ==============================================================================

# ==============================================================================
# SSU_25_NLP_project - data/vector_db.py (v3.1 - 동아리 데이터 추가 최종본)
#
# [변경 사항]
# 1. SQLite DB의 'clubs' 테이블에서 동아리 데이터를 읽어옵니다.
# 2. 동아리 설명글이 길기 때문에 '청킹(Chunking)'을 적용합니다.
# ==============================================================================

# ==============================================================================
# [변경 사항]
# 청킹 중간 결과 저장: DB에서 로드하고 청킹(Chunking)을 완료한 모든 텍스트 청크와 
# 메타데이터를 chunked_data.jsonl 파일로 저장하는 로직이 
# load_source_data 함수 마지막에 추가되었습니다.

# 데이터셋 생성 효율화: 이 JSONL 파일을 사용해 generate_dataset.py에서 
# 불필요한 중복 청킹 과정을 생략하고 바로 데이터를 로드할 수 있게 됩니다.

# [추가된 상수]
# CHUNKED_DATA_PATH = "chunked_data.jsonl": 청킹된 결과물을 저장할 파일 경로.
# ==============================================================================


import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import torch
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json # ⭐ json 임포트 추가 ⭐
import os # ⭐ os 임포트 추가 ⭐

# --- 1. 설정 ---
SOURCE_DB_PATH = "ssu_chatbot_data.db"
VECTOR_DB_PATH = "./chroma_db"  
COLLECTION_NAME = "ssu_knowledge_base" 
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli" 

CHUNK_SIZE = 400     # 청크 글자 수
CHUNK_OVERLAP = 50   # 청크 겹침 글자 수
CHROMA_ADD_BATCH_SIZE = 5000 
# ⭐ 신규 상수: 청킹 결과물을 저장할 JSONL 파일 경로 ⭐
CHUNKED_DATA_PATH = "chunked_data.jsonl" 

# --- 2. 데이터 로드 함수 ---
def load_source_data(db_path):
    print(f"DB 연결 중: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    documents = []
    metadatas = []
    ids = []
    
    # ⭐ 청킹된 데이터를 저장할 리스트 ⭐
    chunk_exports = [] 

    # 청킹 도구
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

    # ---------------------------------------------------------
    # [1] 강의평 로드 (청킹 X)
    # ---------------------------------------------------------
    print(" -> 1/3. 'lecture_reviews' 로딩 중...")
    cursor.execute("SELECT id, subject_name, professor_name, review_text, semester FROM lecture_reviews")
    for row in cursor.fetchall():
        text = f"과목명: {row[1]}, 교수명: {row[2]}, 강의평: {row[3]}"
        documents.append(text)
        metadatas.append({
            "source": "lecture_review",
            "subject": row[1], "professor": row[2], "semester": row[4],
            "original_text": row[3]
        })
        ids.append(f"review_{row[0]}") 
        
        # ⭐ 청킹되지 않은 데이터도 동일한 형식으로 저장 ⭐
        chunk_exports.append({"text": text, "metadata": metadatas[-1]})

    # ---------------------------------------------------------
    # [2] 공지사항 로드 (청킹 O)
    # ---------------------------------------------------------
    print(" -> 2/3. 'notices' 로딩 및 청킹 중...")
    cursor.execute("SELECT id, title, category, full_body_text, link, department FROM notices")
    for row in cursor.fetchall():
        # 원본 데이터와 메타데이터
        original_title, category, full_body, link, department = row[1], row[2], row[3], row[4], row[5]
        
        chunks = text_splitter.split_text(full_body)
        for i, chunk in enumerate(chunks):
            text = f"공지: {original_title} (카테고리: {category})\n내용: {chunk}"
            metadata = {
                "source": "notice",
                "title": original_title, "category": category, "department": department, "link": link,
                "original_text": chunk
            }
            documents.append(text)
            metadatas.append(metadata)
            ids.append(f"notice_{row[0]}_chunk_{i}")
            
            # ⭐ 청킹된 데이터 저장 ⭐
            chunk_exports.append({"text": text, "metadata": metadata})
            
    # ---------------------------------------------------------
    # [3] 동아리 로드 (청킹 O) 
    # ---------------------------------------------------------
    print(" -> 3/3. 'clubs' 로딩 및 청킹 중...")
    try:
        cursor.execute("SELECT id, club_name, category, description, recruitment_info, source_url FROM clubs")
        clubs = cursor.fetchall()
        for row in clubs:
            club_id, club_name, category, description, url = row[0], row[1], row[2], row[3], row[5]

            chunks = text_splitter.split_text(description)
            
            for i, chunk in enumerate(chunks):
                text = f"동아리: {club_name} (분과: {category})\n소개: {chunk}"
                metadata = {
                    "source": "club",
                    "club_name": club_name, 
                    "category": category,
                    "link": url,
                    "original_text": chunk
                }
                documents.append(text)
                metadatas.append(metadata)
                ids.append(f"club_{club_id}_chunk_{i}")
                
                # ⭐ 청킹된 데이터 저장 ⭐
                chunk_exports.append({"text": text, "metadata": metadata})

    except Exception as e:
        print(f"   [경고] 동아리 로드 중 오류: {e}")
            
    conn.close()
    
    # ⭐ 4. 청킹 중간 결과물 저장 로직 추가 ⭐
    print(f"\n[저장] 청킹 결과물 {len(chunk_exports)}개를 '{CHUNKED_DATA_PATH}'에 저장 중...")
    with open(CHUNKED_DATA_PATH, 'w', encoding='utf-8') as f:
        for item in chunk_exports:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("[OK] 청킹 결과물 저장 완료.")

    print(f"\n[OK] 총 {len(documents)}개 청크 로드 완료.")
    return documents, metadatas, ids

# --- 3. 메인 실행 (기존과 동일) ---
def main():
    start_time = time.time()
    
    print("1. 원본 데이터베이스 (SQLite) 로딩 시작...")
    documents, metadatas, ids = load_source_data(SOURCE_DB_PATH)
    
    if not documents: return

    print("\n2. 벡터 데이터베이스 (ChromaDB) 초기화 중...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    # ... (나머지 ChromaDB 로직은 동일) ...
    # (생략)
    
    print("\n2. 벡터 데이터베이스 (ChromaDB) 초기화 중...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print("\n3. 신규 데이터 확인 중...")
    existing_ids = set(collection.get(ids=ids)['ids'])
    
    new_docs, new_metas, new_ids = [], [], []
    for doc, meta, id_str in zip(documents, metadatas, ids):
        if id_str not in existing_ids:
            new_docs.append(doc)
            new_metas.append(meta)
            new_ids.append(id_str)

    if not new_ids:
        print("[완료] 추가할 새로운 데이터가 없습니다.")
        return
        
    print(f" -> 총 {len(documents)}개 중 {len(new_ids)}개의 신규 청크를 처리합니다.")

    print("\n4. BERT 임베딩 모델 로딩 및 변환 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    embeddings = model.encode(new_docs, show_progress_bar=True, batch_size=32)

    print("\n5. 벡터 DB에 저장 중 (배치 처리)...")
    total_items = len(new_ids)
    
    for i in range(0, total_items, CHROMA_ADD_BATCH_SIZE):
        end_idx = min(i + CHROMA_ADD_BATCH_SIZE, total_items)
        print(f" -> 배치를 저장 중... ({end_idx}/{total_items})")
        
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=new_docs[i:end_idx], 
            metadatas=new_metas[i:end_idx], 
            ids=new_ids[i:end_idx]
        )
        
    print("\n[모든 작업 완료]")
    print(f" -> {len(new_ids)}건의 신규 청크 저장 완료.")
    print(f" -> 총 실행 시간: {time.time() - start_time:.2f}초")


if __name__ == "__main__":
    main()