import os
import sys
import json
import sqlite3
import random
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 1. 프로젝트 루트 경로 설정 (부모 폴더의 모듈을 가져오기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. .env 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ssu_chatbot_data.db")

def fetch_documents(limit=10):
    """DB에서 텍스트 데이터를 가져옵니다."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 공지사항
    cur.execute("SELECT title, full_body_text FROM notices WHERE full_body_text IS NOT NULL AND length(full_body_text) > 50")
    notices = [f"[공지] {t}\n{b}" for t, b in cur.fetchall()]
    
    # 강의평
    cur.execute("SELECT subject_name, review_text FROM lecture_reviews WHERE review_text IS NOT NULL AND length(review_text) > 20")
    reviews = [f"[강의평] {s}\n{r}" for s, r in cur.fetchall()]
    
    conn.close()
    
    all_docs = notices + reviews
    random.shuffle(all_docs)
    
    print(f"📊 DB에서 총 {len(all_docs)}개의 문서를 찾았습니다. (최대 {limit}개 사용)")
    return all_docs[:limit]

def generate_qa_pair(text):
    """GPT-4o-mini를 사용하여 질문-정답 쌍 생성"""
    prompt = f"""
    아래 텍스트를 읽고, 챗봇 사용자가 물어볼 만한 자연스러운 '질문'과 그에 대한 '정답'을 1개만 생성해줘.
    정답은 반드시 제공된 텍스트에 있는 내용만 기반해야 해.
    
    [텍스트]:
    {text[:1000]}
    
    [출력 형식 (JSON)]:
    {{
        "question": "생성된 질문",
        "ground_truth": "텍스트 내용을 바탕으로 한 정답"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # ✅ 생성은 저렴한 mini 사용
            messages=[{"role": "system", "content": "너는 데이터셋 생성기야. 반드시 JSON 형식으로만 대답해."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None

if __name__ == "__main__":
    docs = fetch_documents()
    dataset = []
    
    print(f"🚀 gpt-4o-mini로 데이터셋 생성을 시작합니다...")
    
    # ⚠️ 테스트할 때는 range(len(docs)) 대신 range(10)으로 줄여서 먼저 확인하세요!
    for i in tqdm(range(len(docs))):
        qa = generate_qa_pair(docs[i])
        if qa:
            dataset.append(qa)
            
        # 500개마다 중간 저장 (날림 방지)
        if len(dataset) % 500 == 0:
            with open("Evaluation/qa_dataset_intermediate.jsonl", "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 최종 저장
    with open("Evaluation/qa_dataset_5k.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ 생성 완료! 총 {len(dataset)}개 저장됨: Evaluation/qa_dataset_5k.jsonl")