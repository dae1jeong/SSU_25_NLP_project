
# 평가 시 현재 코드 처럼 llm 평가도 유지하고, + ragas (정량적 평가 지표)도 추가하자.
# ragas는 llm을 도구로 활용하여 rag를 자동화된 정량적 평가 지표로 측정함.
# -> 현재 코드보다 좀 더 정밀한 평가.가 가능.

# 주의사항
# rag_pipeline.py 파일에서 415~ Line 주석을 해제해야함.
# 작업자 : 박채은
"""
    ragas_input_path: run_rag_test.py 돌린 결과물 경로
    ragas input 형태
        [
            {
                "question": "...",
                "answer": "...",
                "contexts": [{"id": "...", "text": "...", "meta": {...}}, ...],
                "ground_truths": ["..."]
            },
            ...
        ]
"""

import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    context_precision
)
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

# .env 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
# OpenAI 클라이언트는 LangChain에서 처리하므로 여기서 직접적인 사용은 필요 없음

# ⭐ 1. 상수 정의 및 설정 ⭐
RAGAS_INPUT_PATH = "/Evaluation/data/ragas_qa_dataset.jsonl" 
RAGAS_OUTPUT_CSV = "Evalation/data/ragas_scores.csv"

# ⭐ 2. Ragas 평가자 (LLM Judge) 명시적 설정 ⭐
GPT_4O_LLM = ChatOpenAI(model="gpt-4o", temperature=0) # LLM Judge
OPENAI_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small") # Embedding Model

# --------------------------------------------------------------------------
def load_full_json_data(file_path: str) -> list:
    """단일 JSON 파일(리스트 형태)에서 데이터를 읽어옵니다."""
    if not os.path.exists(file_path):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {file_path}")
        return []

    print(f"-> JSON 파일 로드 시작: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 💡 JSONL이 아닌, 전체 파일을 한 번에 로드 (단일 리스트)
            data = json.load(f) 
        except json.JSONDecodeError as e:
            print(f"오류: JSON 파일 파싱 오류 발생. 오류: {e}")
            return []
    
    print(f"-> 총 {len(data)}개 샘플 로드 완료.")
    return data

def safe_save_csv(df: pd.DataFrame, file_path: str):
    """디렉토리를 확인하고 CSV 파일을 저장하며, 파일명 중복 방지 로직을 사용합니다."""
    
    save_dir = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path)

    # 디렉토리 생성
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"-> 디렉토리 생성: {save_dir}")
    
    # 파일명 중복 방지 로직 (간소화)
    name, ext = os.path.splitext(base_filename)
    counter = 0
    filename_to_save = file_path
    
    while os.path.exists(filename_to_save):
        counter += 1
        new_filename = f"{name}_{counter}{ext}"
        filename_to_save = os.path.join(save_dir, new_filename)
    
    df.to_csv(filename_to_save, index=False)
    print(f"\n✅ 평가 결과가 '{filename_to_save}' 파일로 저장되었습니다.")


def evaluate_ragas_dataset_to_dataframe(ragas_input_data: list) -> pd.DataFrame:

    print("1. Ragas 입력 데이터 전처리 및 매핑 시작...")
    
    # RAGAs가 요구하는 필드 구조
    processed_data = {
        "question": [], 
        "answer": [], 
        "contexts": [],       # 💡 List[str] 형태여야 함
        "ground_truths": []   # 💡 List[str] 형태여야 함
    }
    
    for item in ragas_input_data:
        # 💡 RAGAs 요구 사항에 맞게 필드 매핑
        processed_data["question"].append(item["question"])
        processed_data["answer"].append(item["model_answer"]) # 'answer' 대신 'model_answer' 사용

        # 💡 contexts 필드는 이미 List[str] 형태로 저장된 'context_texts'를 사용
        processed_data["contexts"].append(item["context_texts"])
        
        # 💡 ground_truths는 RAGAs가 List[str]을 요구하므로 단일 문자열을 리스트로 감쌈
        processed_data["ground_truths"].append([item["ground_truth"]]) 
        
    dataset = Dataset.from_dict(processed_data)
    
    print(f"2. 평가 대상 데이터셋 로드 완료. 샘플 수: {len(dataset)}")
    print("-" * 40)
    
    # 3. Ragas 평가 지표 설정
    metrics_to_evaluate = [
        faithfulness, answer_relevancy, context_recall, context_precision
    ]
    
    print("3. Ragas 평가 시작 (LLM Judge: GPT-4o 사용)...")
    
    # ⭐ 4. Ragas 평가 실행 (LLM과 Embeddings 모델 명시적으로 전달) ⭐
    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_evaluate,
        llm=GPT_4O_LLM,
        embeddings=OPENAI_EMBEDDINGS 
    )
    
    print("4. 평가 완료.")
    print("-" * 40)

    # 5. 결과를 Pandas DataFrame으로 변환하여 반환
    result_df = result.to_pandas()
    
    print("5. 최종 요약 점수:")
    print(result)
    print("\n✅ 샘플별 상세 평가 결과 DataFrame 반환.")
    
    return result.to_pandas()

# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    # 💡 RAGAs 입력 경로를 직전에 저장한 FULL JSON 파일 경로로 변경
    # RAGAS_INPUT_PATH = "/Evaluation/data/ragas_qa_dataset.jsonl" 
    RAGAS_INPUT_PATH = "Evaluation/data/rag_evaluation_results_full.json" 
    
    # 1. JSON 파일 로드 (함수 이름도 load_full_json_data로 변경)
    ragas_data = load_full_json_data(RAGAS_INPUT_PATH)
    
    if not ragas_data:
        print("로드된 데이터가 없어 Ragas 평가를 종료합니다.")
    else:
        # 2. 평가 실행 및 DataFrame으로 결과 저장
        print(f"총 {len(ragas_data)}개 샘플로 Ragas 평가를 시작합니다.")
        evaluation_dataframe = evaluate_ragas_dataset_to_dataframe(ragas_data)
        
        # 3. 결과를 CSV 파일로 저장 (파일명 중복 방지 및 디렉토리 자동 생성)
        safe_save_csv(evaluation_dataframe, RAGAS_OUTPUT_CSV)
        
        print("\n[ 최종 평가 결과 DataFrame ]")
        print(evaluation_dataframe.head())