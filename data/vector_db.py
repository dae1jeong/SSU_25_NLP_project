#data.py를 통해 크롤링한 데이터를 1차적으로 전처리하고 처리한 데이터 베이스에서 데이트를 로드
#SentenceTransformer 모델을 이용해 모든 텍스트를 벡터로 변환
#변환된 벡터와 원본 테스트의 메타데이터를 벡터 db에 저장





# ==============================================================================
# SSU_25_NLP_project - build_vector_db.py (v2.2 - ChromaDB 배치 저장 오류 수정)
#
# 역할: RAG 챗봇의 '검색 엔진(Retriever)' 역할을 할 벡터 DB를 구축.
#
# [v2.2 변경 사항]
# 1. (ChromaDB) 1만 건 이상의 데이터를 한 번에 .add() 할 때 발생하는 
#               InternalError (Batch size > max batch size)를 해결.
# 2. (ChromaDB) 데이터를 5000개씩 묶어서 반복문으로 나눠 저장하도록 수정.
# ==============================================================================

import sqlite3
import chromadb # 벡터 DB
from sentence_transformers import SentenceTransformer # BERT 임베딩 모델
from tqdm import tqdm # 진행률 표시
import time
import torch
import math # (수정) 배치 계산을 위해 import

# (수정) 청킹을 위한 라이브러리 임포트 (langchain -> langchain_text_splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. 설정 ---
SOURCE_DB_PATH = "ssu_chatbot_data.db"
VECTOR_DB_PATH = "./chroma_db"  
COLLECTION_NAME = "ssu_knowledge_base" 
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli" 

CHUNK_SIZE = 400     # 청크 글자 수
CHUNK_OVERLAP = 50   # 청크 겹침 글자 수

# (수정) ChromaDB에 저장할 때 사용할 배치 크기 (오류난 5461보다 작게 설정)
CHROMA_ADD_BATCH_SIZE = 5000


# --- 2. 원본 DB에서 텍스트 데이터 로드 (청킹 로직 추가) ---
def load_source_data(db_path):
    """
    ssu_chatbot_data.db에서 강의평과 공지사항 데이터를 읽어옵니다.
    (수정) 공지사항은 청킹을 수행합니다.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    documents = []  # 임베딩할 텍스트 (청크)
    metadatas = []  # 각 청크의 부가정보 (필터링용)
    ids = []        # 각 청크의 고유 ID

    # --- 1. 강의평 로드 (청킹 X) ---
    print(" -> 1/2. 'lecture_reviews' 테이블 로딩 중... (청킹 안함)")
    cursor.execute("SELECT id, subject_name, professor_name, review_text, semester FROM lecture_reviews")
    reviews = cursor.fetchall()
    for row in reviews:
        text_to_embed = f"과목명: {row[1]}, 교수명: {row[2]}, 강의평: {row[3]}"
        
        documents.append(text_to_embed)
        metadatas.append({
            "source": "lecture_review",
            "subject": row[1],      # NLU 필터링용
            "professor": row[2],    # NLU 필터링용
            "semester": row[4],
            "original_text": row[3] # LLM에게 넘겨줄 원본
        })
        ids.append(f"review_{row[0]}") 

    # --- 2. 공지사항 로드 (청킹 O) ---
    print(" -> 2/2. 'notices' 테이블 로딩 및 청킹 중...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len # 글자 수 기반
    )
    
    cursor.execute("SELECT id, title, category, full_body_text, link, department FROM notices")
    notices = cursor.fetchall()
    
    for row in notices:
        notice_id = row[0]
        title = row[1]
        category = row[2]
        full_text = row[3]
        link = row[4]
        department = row[5]
        
        chunks = text_splitter.split_text(full_text)
        
        for i, chunk_text in enumerate(chunks):
            text_to_embed = f"공지 제목: {title} (카테고리: {category})\n본문: {chunk_text}"
            
            documents.append(text_to_embed)
            metadatas.append({
                "source": "notice",
                "title": title,
                "category": category,      # NLU 필터링용
                "department": department,  # NLU 필터링용
                "link": link,
                "original_text": chunk_text # LLM에게 넘겨줄 원본 (청크)
            })
            ids.append(f"notice_{notice_id}_chunk_{i}")
        
    conn.close()
    print(f"\n[OK] 원본 DB 로드 및 청킹 완료. (총 {len(documents)}개 청크 생성)")
    return documents, metadatas, ids

# --- 3. 메인 파이프라인 ---
def main():
    start_time = time.time()
    
    # 1. 원본 데이터 로드 (청킹 포함)
    print("1. 원본 데이터베이스 (SQLite) 로딩 및 청킹 시작...")
    documents, metadatas, ids = load_source_data(SOURCE_DB_PATH)
    
    if not documents:
        print("[완료] 임베딩할 데이터가 없습니다. data.py를 먼저 실행하세요.")
        return

    # 2. 벡터 DB(Chroma) 초기화
    print("\n2. 벡터 데이터베이스 (ChromaDB) 초기화 중...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )

    # 3. 이미 처리된 데이터 확인 (중복 방지)
    print("\n3. 이미 처리된 데이터 확인 중...")
    existing_ids = set(collection.get(ids=ids)['ids']) 
    
    if len(existing_ids) == len(ids):
        print(f"[완료] 모든 데이터 ({len(ids)}개 청크)가 이미 벡터 DB에 저장되어 있습니다.")
        end_time = time.time()
        print(f"총 실행 시간: {end_time - start_time:.2f}초")
        return
        
    # 새로 추가해야 할 데이터만 필터링
    new_documents = []
    new_metadatas = []
    new_ids = []
    
    for doc, meta, id_str in zip(documents, metadatas, ids):
        if id_str not in existing_ids:
            new_documents.append(doc)
            new_metadatas.append(meta)
            new_ids.append(id_str)

    if not new_ids:
        print("[완료] 새로운 데이터가 없습니다. 벡터 DB가 최신 상태입니다.")
        return
        
    print(f" -> 총 {len(documents)}개 청크 중 {len(new_ids)}개의 신규 청크를 처리합니다.")

    # 4. BERT 임베딩 모델 로드
    print("\n4. BERT 임베딩 모델 로딩 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f" -> 모델 로드 완료. (사용 장치: {model.device})")

    # 5. 텍스트 임베딩 (벡터화)
    print(f"\n5. 텍스트 임베딩 변환 작업 시작... (총 {len(new_ids)}건)")
    embeddings = model.encode(
        new_documents, 
        show_progress_bar=True, 
        batch_size=32 
    )
    print(" -> 임베딩 변환 완료.")

    # 6. 벡터 DB에 저장 (수정: 배치 처리)
    print("\n6. 벡터 DB에 임베딩 저장 중 (배치 처리)...")
    
    total_items = len(new_ids)
    total_batches = math.ceil(total_items / CHROMA_ADD_BATCH_SIZE)
    
    for i in range(0, total_items, CHROMA_ADD_BATCH_SIZE):
        print(f" -> {i//CHROMA_ADD_BATCH_SIZE + 1}/{total_batches} 배치를 저장 중...")
        
        # 1. 현재 배치의 시작과 끝 인덱스 계산
        start_idx = i
        end_idx = min(i + CHROMA_ADD_BATCH_SIZE, total_items)
        
        # 2. 모든 리스트와 배열을 동일하게 슬라이싱
        batch_ids = new_ids[start_idx:end_idx]
        batch_documents = new_documents[start_idx:end_idx]
        batch_metadatas = new_metadatas[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]
        
        # 3. 슬라이싱된 배치(묶음)를 .add()
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents, 
            metadatas=batch_metadatas, 
            ids=batch_ids              
        )

    end_time = time.time()
    print("\n[모든 작업 완료]")
    print(f" -> {len(new_ids)}건의 신규 청크가 벡터 DB에 성공적으로 저장되었습니다.")
    print(f" -> 총 실행 시간: {end_time - start_time:.2f}초")
    print(f" -> 벡터 DB는 '{VECTOR_DB_PATH}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()