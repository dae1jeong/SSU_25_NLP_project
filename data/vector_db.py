#data.py를 통해 크롤링한 데이터를 1차적으로 전처리하고 처리한 데이터 베이스에서 데이트를 로드
#SentenceTransformer 모델을 이용해 모든 텍스트를 벡터로 변환
#변환된 벡터와 원본 테스트의 메타데이터를 벡터 db에 저장




import sqlite3
import chromadb # 벡터 DB
from sentence_transformers import SentenceTransformer # BERT 임베딩 모델
from tqdm import tqdm 
import time
import torch 

#  설정 
SOURCE_DB_PATH = "ssu_chatbot_data.db"
VECTOR_DB_PATH = "./chroma_db"  # 벡터 DB를 저장할 폴더 경로
COLLECTION_NAME = "ssu_knowledge_base" # 벡터를 저장할 컬렉션 이름

#  BERT 임베딩 모델 

EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli" 


# 원본 DB에서 텍스트 데이터 로드 
def load_source_data(db_path):
    """
    ssu_chatbot_data.db에서 강의평과 공지사항 데이터를 읽어옵니다.
    [반환값]
    - documents: 임베딩할 텍스트 리스트
    - metadatas: 각 텍스트의 부가정보 리스트 (원본 식별용)
    - ids: 각 텍스트의 고유 ID 리스트
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    documents = []
    metadatas = []
    ids = []
    
    # 강의평 로드
    print(" -> 1/2. 'lecture_reviews' 테이블에서 데이터 로딩 중...")
    cursor.execute("SELECT id, subject_name, professor_name, review_text, semester FROM lecture_reviews")
    reviews = cursor.fetchall()
    for row in reviews:
        
        text_to_embed = f"과목: {row[1]} ({row[2]} 교수) 강의평: {row[3]}"
        
        documents.append(text_to_embed)
        metadatas.append({
            "source": "lecture_review", 
            "subject": row[1],
            "professor": row[2],
            "semester": row[4],
            "original_text": row[3] # 나중에 LLM에게 넘겨줄 원본 텍스트
        })
        # ChromaDB는 ID가 문자열이어야 하며, 중복되면 안 됨
        ids.append(f"review_{row[0]}") 

    #  공지사항 로드
    print(" -> 2/2. 'notices' 테이블에서 데이터 로딩 중...")
    cursor.execute("SELECT id, title, category, full_body_text, link FROM notices")
    notices = cursor.fetchall()
    for row in notices:
        
        text_to_embed = f"공지: {row[1]} (카테고리: {row[2]})\n{row[3]}"

        documents.append(text_to_embed)
        metadatas.append({
            "source": "notice", # 출처
            "title": row[1],
            "category": row[2],
            "link": row[4],
            "original_text": row[3] # LLM에게 넘겨줄 원본
        })
        # 강의평 ID와 겹치지 않게 'notice_' 접두사 추가
        ids.append(f"notice_{row[0]}")
        
    conn.close()
    print(f"\n[OK] 원본 DB에서 총 {len(documents)}건의 텍스트 로드 완료.")
    return documents, metadatas, ids

# 메인 파이프라인 
def main():
    start_time = time.time()
    
    #  원본 데이터 로드
    print("1. 원본 데이터베이스 (SQLite) 로딩 시작...")
    documents, metadatas, ids = load_source_data(SOURCE_DB_PATH)
    
    if not documents:
        print("[완료] 임베딩할 데이터가 없습니다. data.py를 먼저 실행하세요.")
        return

    # 2. 벡터 DB 초기화
    # PersistentClient: 데이터를 디스크에 저장
    print("\n2. 벡터 데이터베이스 (ChromaDB) 초기화 중...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # 컬렉션 가져오기. 없으면 새로 생성.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용 
    )

    
    # 스크립트를 여러 번 실행해도 중복 처리를 방지 
    print("\n3. 이미 처리된 데이터 확인 중...")
    existing_ids = set(collection.get(ids=ids)['ids']) # 이미 있는 ID 목록 조회
    
    if len(existing_ids) == len(ids):
        print(f"[완료] 모든 데이터 ({len(ids)}건)가 이미 벡터 DB에 저장되어 있습니다.")
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
        
    print(f" -> 총 {len(documents)}건 중 {len(new_ids)}건의 신규 데이터를 처리합니다.")

    # 4. BERT 임베딩 모델 로드
    print("\n4. BERT 임베딩 모델 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)")
    
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f" -> 모델 로드 완료. (사용 장치: {model.device})")

    # 5. 텍스트 임베딩 
    
    print(f"\n5. 텍스트 임베딩 변환 작업 시작... (총 {len(new_ids)}건)")
    embeddings = model.encode(
        new_documents, 
        show_progress_bar=True, # 진행률 표시
        batch_size=32 
    )
    print(" -> 임베딩 변환 완료.")

    # 6. 벡터 DB에 저장
    # ChromaDB는 대용량 데이터를 자동으로 나눠서 저장해줍니다.
    print("\n6. 벡터 DB에 임베딩 저장 중...")
    
    
    collection.add(
        embeddings=embeddings,
        documents=new_documents, # 원본 텍스트도 함께 저장 
        metadatas=new_metadatas, 
        ids=new_ids              
    )

    end_time = time.time()
    print("\n[모든 작업 완료]")
    print(f" -> {len(new_ids)}건의 신규 데이터가 벡터 DB에 성공적으로 저장되었습니다.")
    print(f" -> 총 실행 시간: {end_time - start_time:.2f}초")
    print(f" -> 벡터 DB는 '{VECTOR_DB_PATH}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()



