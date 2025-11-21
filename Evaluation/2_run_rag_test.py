import json
import random
import sys
import os
from tqdm import tqdm

# RAG 파이프라인 가져오기
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG.rag_pipeline import RAGPipeline, call_openai_api

def run_test():
    data_path = "Evaluation/qa_dataset_5k.jsonl"
    if not os.path.exists(data_path):
        print("❌ 데이터셋 파일이 없습니다. 1번 코드를 먼저 실행하세요.")
        return

    # 1. 데이터셋 로드
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    # 2. 100개 랜덤 샘플링 (비용 절약)
    sample_size = min(100, len(all_data))
    test_samples = random.sample(all_data, sample_size)
    
    print(f"🧪 {sample_size}개 문제로 RAG 성능 테스트를 시작합니다.")
    print("   (참고: rag_pipeline.py에 설정된 모델을 사용합니다)")

    # 3. RAG 엔진 로딩
    rag = RAGPipeline()
    results = []
    
    # 4. 문제 풀기
    for item in tqdm(test_samples):
        question = item['question']
        ground_truth = item['ground_truth']
        
        # 챗봇에게 질문 던지기
        try:
            predicted_answer = rag.answer_with_llm(question, llm_call=call_openai_api)
        except Exception as e:
            predicted_answer = f"에러 발생: {e}"
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer
        })

    # 5. 결과 저장
    with open("Evaluation/rag_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("✅ 실행 완료! 결과 파일: Evaluation/rag_test_results.json")

if __name__ == "__main__":
    run_test()