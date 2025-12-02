import json
import numpy as np

# ======================
# 이 세 가지 지표는 챗봇이 사용자 질문에 대한 정답 문서를 얼마나 정확하고 
# 빠르게 상위 순위에 배치했는지 평가하는 데 초점을 맞춥니다.
# 1. Recall@K는 시스템이 제시한 상위 K개의 검색 결과 목록 안에 정답 항목이 포함되어 있는지 
# 여부를 측정하는 지표입니다. 
# 2. MMR(Mean Reciprocal Rank, 평균 역순위)은 검색 결과 목록에서 
# 첫 번째로 나타난 정답 항목의 순위가 얼마나 높은지를 측정합니다. 
# MRR은 정답을 1위에 배치하는 시스템(1.0점 획득)이 5위(0.2점 획득)에 배치하는 시스템보다 훨씬 우수하다고 평가합니다. 
# 3. RRF (Reciprocal Rank Fusion, 역순위 융합)는 
# 여러 독립적인 검색 시스템의 순위 목록을 효과적으로 결합하여 
# 최종 순위를 결정하기 위해 사용되는 점수 체계입니다. 
# 작업자 : 박채은

def calculate_retrieval_metrics_with_latency(results_path: str, K: int = 5):
    """
    저장된 JSON 결과 파일을 로드하여 Recall@K, MRR, RRF 및 평균 레이턴시를 계산합니다.
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 지표 계산용 리스트 초기화
    recall_scores = []
    mrr_scores = []
    rrf_scores = [] 
    latency_scores = [] # 💡 레이턴시 측정을 위한 리스트 추가
    
    # RRF 계산을 위한 상수
    k_rrf = 60 

    for item in data:
        # 1. 필수 데이터 추출
        gt_id = item.get('ground_truth_chunk_id')
        retrieved_list = item.get('retrieved_chunks', [])
        
        # 💡 레이턴시 값 추출 (RAG flow가 아닌 경우는 0으로 처리)
        latency = item.get('latency_seconds', 0.0)
        
        # RAG flow를 탄 경우에만 검색 지표 및 레이턴시를 측정 대상으로 삼음
        if item.get('is_rag_flow', False):
            latency_scores.append(latency) # 💡 레이턴시 점수 누적

            # 2. 정답 청크의 순위(Rank) 찾기
            found_rank = 0
            for index, chunk in enumerate(retrieved_list):
                if chunk.get('id') == gt_id:
                    found_rank = index + 1
                    break
            
            # 3. 검색 지표 계산
            recall_scores.append(1 if (found_rank > 0 and found_rank <= K) else 0)
            mrr_scores.append(1.0 / found_rank if found_rank > 0 else 0.0)
            rrf_scores.append(1.0 / (k_rrf + found_rank) if found_rank > 0 else 0.0)


    # 4. 최종 평균 계산 및 결과 통합
    metrics = {
        # 💡 전체 평균 레이턴시 추가
        "Average Latency (sec)": np.mean(latency_scores) if latency_scores else 0.0,
        
        # 검색 정확도 지표
        f"Mean Recall@{K}": np.mean(recall_scores) if recall_scores else 0.0,
        "MRR (Mean Reciprocal Rank)": np.mean(mrr_scores) if mrr_scores else 0.0,
        f"Mean RRF (k={k_rrf})": np.mean(rrf_scores) if rrf_scores else 0.0
    }
    
    return metrics

# ------------------------------------------------------------------
# 사용 예시
# ------------------------------------------------------------------
results_file = "Evaluation/data/rag_evaluation_results_full.json"
retrieval_metrics = calculate_retrieval_metrics_with_latency(results_file, K=5)
print(retrieval_metrics)