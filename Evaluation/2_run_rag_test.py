import json
import random
import sys
import os
from tqdm import tqdm

from dataclasses import dataclass # 💡 dataclasses 임포트 추가
from typing import List, Dict     # 💡 typing 임포트 추가 (List, Dict 사용을 위해)

# RAG 파이프라인 가져오기
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# RAG.rag_pipeline_chunked.py 파일에서 직접 임포트할 수 없는 경우, 아래 클래스들을 여기에 정의합니다.
from RAG.rag_pipeline_chunked import RAGPipeline, call_openai_api 


# -----------------------------------------------
# 💡 [필수 추가] ChunkDocument 재정의
# -----------------------------------------------
@dataclass
class ChunkDocument:
    id: str
    text: str
    meta: Dict
    tokens: List[str]


# -----------------------------------------------
# 💡 [필수 추가] EvaluationResult 정의
# -----------------------------------------------
@dataclass
class EvaluationResult:
    """RAG 시스템 평가에 필요한 모든 결과를 담는 구조"""
    query: str                       
    model_answer: str                
    retrieved_chunks: List[ChunkDocument] # 💡 ChunkDocument 사용
    context_texts: List[str]         
    is_rag_flow: bool


def run_test():
    data_path = "Evaluation/data/ragas_qa_dataset_remove2.jsonl"
    if not os.path.exists(data_path):
        print("❌ 데이터셋 파일이 없습니다. 1번 코드를 먼저 실행하세요.")
        return

    # 1. 데이터셋 로드
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
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
        # 💡 Recall/MRR/RRF 평가를 위해 ground_truth_chunk_id를 가져옵니다.
        # 이 필드가 없으면 Recall/MRR/RRF는 계산 불가능합니다.
        ground_truth_chunk_id = item.get('ground_truth_chunk_id')
        
        # 챗봇에게 질문 던지기
        try:
            # 💡 평가 전용 함수 호출
            eval_result: EvaluationResult = rag.answer_with_llm_EVAL(
                question, 
                llm_call=call_openai_api
            )
        except Exception as e:
            # 에러 발생 시 EvaluationResult 형태로 저장
            eval_result = EvaluationResult(
                query=question,
                model_answer=f"에러 발생: {e}",
                retrieved_chunks=[],
                context_texts=[],
                is_rag_flow=False
            )
        
        # 💡 EvaluationResult 객체를 JSON 저장을 위한 딕셔너리로 변환
        
        # ChunkDocument 객체는 JSON으로 바로 저장이 안 되므로 딕셔너리로 변환합니다.
        retrieved_chunks_data = [
            {
                "id": d.id, 
                "text": d.text, 
                "meta": d.meta
            } 
            for d in eval_result.retrieved_chunks
        ]
        
        # 모든 평가 지표 계산에 필요한 원천 데이터를 results에 저장
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_chunk_id": ground_truth_chunk_id, # Recall/MRR/RRF 기준점
            
            # --- 모델의 출력 및 검색 결과 ---
            "model_answer": eval_result.model_answer,
            "is_rag_flow": eval_result.is_rag_flow,
            "retrieved_chunks": retrieved_chunks_data, 
            "context_texts": eval_result.context_texts, # RAGAs 입력용 텍스트 목록
            "latency_seconds": eval_result.latency_seconds # 💡 여기 추가!
        })

    # 5. 결과 저장
    # 💡 결과 파일 이름을 명확하게 변경합니다.
    output_filename = "Evaluation/data/rag_evaluation_results_full.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"실행 완료! 결과 파일: {output_filename}")

if __name__ == "__main__":
    run_test()