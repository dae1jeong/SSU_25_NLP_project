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
# # 1. (Load) SQLite DB에서 3가지 데이터(강의평, 공지사항, 동아리)를 로드합니다.
# # 2. (Chunking) 긴 텍스트(공지사항, 동아리)는 검색 정확도를 위해 작은 단위로 쪼갭니다.
# -> chunked_db.py의 결과인 chunked_data.jsonl

# ------> 여기서부터 시작
# 3. (Embedding) SBERT 모델을 사용해 텍스트를 768차원 숫자 벡터로 변환합니다.
# 4. (Save) 벡터와 메타데이터를 ChromaDB에 저장합니다. (대용량 배치 처리 포함)
# ==============================================================================


# ======================================================================
# vector_db.py
#
# 역할: chunked_data.jsonl에서 청킹된 텍스트를 로드하고
#       SentenceTransformer를 이용해 임베딩 후 ChromaDB에 저장
# ======================================================================

import json
import os
import time
import torch
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNKED_DATA_PATH = os.path.join(PROJECT_ROOT, "chunked_data.jsonl")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "ssu_knowledge_base"
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"
CHROMA_ADD_BATCH_SIZE = 5000  # 대용량 배치 처리

def load_chunked_data(jsonl_path):
    chunked_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunked_data.append(item)
    print(f"[OK] {len(chunked_data)}개의 청킹 데이터를 로드 완료.")
    return chunked_data

def main():
    start_time = time.time()
    
    # 1. JSONL 로드
    if not os.path.exists(CHUNKED_DATA_PATH):
        print(f"[에러] chunked_data.jsonl 파일이 없습니다: {CHUNKED_DATA_PATH}")
        return
    chunked_data = load_chunked_data(CHUNKED_DATA_PATH)
    
    if not chunked_data:
        print("[완료] 로드된 데이터가 없습니다.")
        return
    
    # 2. ChromaDB 초기화
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. 신규 데이터 확인
    ids = [item["id"] for item in chunked_data]
    existing_result = collection.get(ids=ids, include=[])
    existing_ids = set(existing_result.get('ids', []))
    
    new_docs, new_metas, new_ids = [], [], []
    for item in chunked_data:
        if item["id"] not in existing_ids:
            new_docs.append(item["text"])
            new_metas.append(item["metadata"])
            new_ids.append(item["id"])
    
    if not new_ids:
        print("[완료] 추가할 신규 데이터가 없습니다.")
        return
    
    print(f" -> 총 {len(chunked_data)}개 중 {len(new_ids)}개의 신규 청크를 처리합니다.")
    
    # 4. BERT 임베딩
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    embeddings = model.encode(new_docs, show_progress_bar=True, batch_size=32)
    
    # 5. ChromaDB 저장 (배치 처리)
    total_items = len(new_ids)
    for i in range(0, total_items, CHROMA_ADD_BATCH_SIZE):
        end_idx = min(i + CHROMA_ADD_BATCH_SIZE, total_items)
        print(f" -> 배치 저장 중 ({end_idx}/{total_items})")
        collection.add(
            embeddings=embeddings[i:end_idx].tolist(),
            documents=new_docs[i:end_idx],
            metadatas=new_metas[i:end_idx],
            ids=new_ids[i:end_idx]
        )
    
    print("\n[모든 작업 완료]")
    print(f" -> {len(new_ids)}건의 신규 청크 저장 완료.")
    print(f" -> 총 실행 시간: {time.time() - start_time:.2f}초")

if __name__ == "__main__":
    main()
