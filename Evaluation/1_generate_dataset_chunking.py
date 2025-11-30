# ==============================================================================
# SSU_25_NLP_project - data/generate_dataset.py (v1.1 - RAG 일관성 확보)
#
# [변경 목적]
#
# 합성 데이터셋 생성 시 컨텍스트(Context)를 실제 RAG 챗봇의 검색 결과와 동일하게 구성하여,
# RAG 평가 지표의 **실제 적용 가능성(Realism)**을 확보합니다.
# vector_db.py의 청킹 전략(400자 청크, 50자 겹침)과 완전히 일치시킵니다.
#
# [주요 변경 사항]
#
# 청킹 로직 통합: generate_dataset.py의 load_all_data 함수 내에서 vector_db.py와
# 동일한 설정의 **RecursiveCharacterTextSplitter**를 사용하여 DB에서 로드된 원문에 청킹을 적용합니다.
#
# 데이터 소스별 처리:
#
# 강의평 (lecture_reviews): 원문 전체를 하나의 컨텍스트로 사용합니다. (청킹 제외).
#
# 공지사항 (notices) 및 동아리 (clubs): 긴 원문을 CHUNK_SIZE=400, CHUNK_OVERLAP=50을 사용하여
# 청킹한 후, 각 청크를 독립적인 컨텍스트로 사용하여 QA 쌍을 생성합니다.
#
# [효과]
#
# 합성 데이터셋의 contexts 필드에 담기는 문서 조각들이 실제 챗봇이 검색하여 LLM에 전달하는 조각과 일치하므로,
# 생성된 QA 쌍이 RAG 시스템의 실제 동작을 더 정확하게 반영합니다.
# ==============================================================================

# ==============================================================================
# chorma_db.py를 실행 시 생긴 중간 결과물 jsonl을 사용해 generate dataset.함.
#
# ==============================================================================





import os
import json
import sqlite3
import random
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


# vector_db.py가 생성한 파일명과 일치시켜야 함
CHUNKED_DATA_PATH = "chunked_data.jsonl" 



# ---------------------------
# 환경 설정
# ---------------------------
load_dotenv()
DB_PATH = "ssu_chatbot_data.db"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", " "]
)

# ---------------------------
# DB에서 데이터 로드
# ---------------------------
def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    lecture_reviews = []
    notices_chunks = []
    clubs_chunks = []

    # --- lecture_reviews (청킹 X)
    cursor.execute("SELECT id, subject_name, professor_name, review_text, semester FROM lecture_reviews")
    for row in cursor.fetchall():
        text = f"과목명: {row[1]}, 교수명: {row[2]}, 강의평: {row[3]}"
        lecture_reviews.append(text)

    # --- notices (청킹 O)
    cursor.execute("SELECT id, title, category, full_body_text FROM notices")
    for row in cursor.fetchall():
        chunks = text_splitter.split_text(row[3])
        for chunk in chunks:
            text = f"공지: {row[1]} (카테고리: {row[2]})\n내용: {chunk}"
            notices_chunks.append(text)

    # --- clubs (청킹 O)
    cursor.execute("SELECT id, club_name, category, description FROM clubs")
    for row in cursor.fetchall():
        chunks = text_splitter.split_text(row[3])
        for chunk in chunks:
            text = f"동아리: {row[1]} (분과: {row[2]})\n소개: {chunk}"
            clubs_chunks.append(text)

    conn.close()
    return lecture_reviews, notices_chunks, clubs_chunks

# ---------------------------
# 청킹된 데이터 로드 함수
# ---------------------------
# DB 로드 함수 말고 이걸 사용하도록 main에서 코드 수정 필요
def load_all_data_from_chunks():
    """
    DB에서 직접 로드 및 청킹하는 대신, vector_db.py가 생성한 JSONL 파일에서 청킹된 데이터를 로드합니다.
    """
    lecture_reviews = []
    notices_chunks = []
    clubs_chunks = []

    if not os.path.exists(CHUNKED_DATA_PATH):
        print(f"오류: 청킹 결과 파일 '{CHUNKED_DATA_PATH}'을 찾을 수 없습니다. vector_db.py를 먼저 실행하세요.")
        return lecture_reviews, notices_chunks, clubs_chunks

    print(f"-> 청킹 결과 파일 로드 중: {CHUNKED_DATA_PATH}")
    with open(CHUNKED_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="청크 데이터 로드"):
            item = json.loads(line.strip())
            text = item['text']
            source = item['metadata']['source']
            chunk_id = item['id']  # 수정된 부분....... text, chunk_id를 함께 반환하도록.

            if source == "lecture_review":
                lecture_reviews.append((text, chunk_id))
            elif source == "notice":
                notices_chunks.append((text, chunk_id))
            elif source == "club":
                clubs_chunks.append((text, chunk_id))
    
    return lecture_reviews, notices_chunks, clubs_chunks


# ---------------------------
# 랜덤 샘플링 (비율 맞춤)
# ---------------------------
def sample_documents(lecture_reviews, notices_chunks, clubs_chunks, NUM_QA=10, ratios=(2,5,3)):
    total_ratio = sum(ratios)
    num_lecture = round(NUM_QA * ratios[0] / total_ratio)
    num_notice = round(NUM_QA * ratios[1] / total_ratio)
    num_club = NUM_QA - num_lecture - num_notice

    sampled_docs = []
    if lecture_reviews:
        sampled_docs += random.sample(lecture_reviews, min(num_lecture, len(lecture_reviews)))
    if notices_chunks:
        sampled_docs += random.sample(notices_chunks, min(num_notice, len(notices_chunks)))
    if clubs_chunks:
        sampled_docs += random.sample(clubs_chunks, min(num_club, len(clubs_chunks)))

    random.shuffle(sampled_docs)
    return sampled_docs

# ---------------------------
# QA 생성
# ---------------------------
def generate_qa_pair(text):
    prompt = f"""
    아래 텍스트를 읽고, 챗봇 사용자가 물어볼 만한 자연스러운 질문과 그에 대한 정답 1개를 생성해줘.
    정답은 반드시 제공된 텍스트에 있는 내용만 기반해야 합니다.

    [텍스트]:
    {text[:1000]}

    [출력 JSON]:
    {{
        "question": "생성된 질문",
        "ground_truth": "텍스트 내용을 바탕으로 한 정답"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 데이터셋 생성기야. 반드시 JSON 형식으로만 대답해."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7  # 창의적인 정도
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None

# ---------------------------
# 메인 실행
# ---------------------------
if __name__ == "__main__":
    NUM_QA = 10    #질문 수 저장
    lecture_reviews, notices_chunks, clubs_chunks = load_all_data_from_chunks() # load_all_data_from_chunks()
    sampled_docs = sample_documents(lecture_reviews, notices_chunks, clubs_chunks, NUM_QA=NUM_QA, ratios=(2,5,3))

    dataset = []
    for doc_tuple in tqdm(sampled_docs, desc="QA 생성중"):
        text, chunk_id = doc_tuple # 💡 수정 2: 튜플에서 text와 chunk_id를 분리
        qa = generate_qa_pair(text)
        if qa:
            # 💡 수정 3: 생성된 QA 쌍에 chunk_id를 추가
            qa['ground_truth_chunk_id'] = chunk_id
            dataset.append(qa)

    os.makedirs("Evaluation/data", exist_ok=True)
    output_file = "Evaluation/data/ragas_qa_dataset2.jsonl"  #OUTPUT 경로
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"QA 생성 완료! 총 {len(dataset)}개 저장됨 → {output_file}")
