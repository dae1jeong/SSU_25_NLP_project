#  rag_test_results_1129_*
# 1~11까지 blue score print
import json
import sacrebleu
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# 파일 경로 설정 (현재 상황에 맞게 수정하세요)
BASE_PATH = "Evaluation/data/"
FILE_PREFIX = "rag_test_results_1129_"
NUM_FILES = 11

def _load_data_and_extract(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """단일 JSON 파일을 로드하여 모델 답변과 정답을 추출합니다."""
    
    if not os.path.exists(file_path):
        return [], []
    
    hypotheses = []
    references = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            for item in data:
                model_answer = item.get('model_answer')
                ground_truth = item.get('ground_truth')
                
                if model_answer and ground_truth:
                    hypotheses.append(model_answer)
                    # SacreBLEU는 참조 문장을 List[List[str]] 형태로 요구합니다.
                    references.append([ground_truth]) 
        except json.JSONDecodeError:
            print(f"❌ 오류: {file_path} 파일 JSON 파싱 오류 발생.")
            return [], []
            
    return hypotheses, references

def calculate_all_files_individually():
    """1번부터 NUM_FILES까지 순회하며 파일별 코퍼스 BLEU 점수를 계산합니다."""
    
    individual_scores = {}
    
    print(f"📄 총 {NUM_FILES}개 파일의 BLEU 점수를 개별 계산합니다.")

    for i in tqdm(range(1, NUM_FILES + 1), desc="파일별 BLEU 계산 중"):
        file_name = f"{FILE_PREFIX}{i}.json"
        file_path = os.path.join(BASE_PATH, file_name)
        
        hypotheses, references = _load_data_and_extract(file_path)
        total_samples = len(hypotheses)
        
        if total_samples == 0:
            individual_scores[file_name] = {"BLEU-4 Score": 0.0, "Samples": 0}
            continue

        # SacreBLEU 코퍼스 BLEU 계산
        # list(zip(*references))를 사용하여 입력 형태를 맞춥니다.
        bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))
        
        individual_scores[file_name] = {
            "BLEU-4 Score": round(bleu.score, 4), 
            "Samples": total_samples
        }
        
    return individual_scores

# ==============================================================================
# 🚀 실행 부분
# ==============================================================================
if __name__ == "__main__":
    
    # 1. 파일별 BLEU 점수 계산
    scores = calculate_all_files_individually()
    
    # 2. 결과 출력 (요청하신 파일명 / 스코어 형태)
    print("\n" + "="*50)
    print("         📋 개별 파일 코퍼스 BLEU 평가 결과 📋           ")
    print("="*50)
    print(f"{'파일명':<30} | {'BLEU-4 Score':<15} | {'Samples':<8}")
    print("-" * 50)
    
    # 결과를 파일명 순서대로 출력
    for filename, result in sorted(scores.items()):
        score_str = f"{result['BLEU-4 Score']:.4f}"
        print(f"{filename:<30} | {score_str:<15} | {result['Samples']:<8}")
        
    print("="*50)

#          📋 최종 BLEU-4 평가 결과 📋
# ==============================================
# Total Samples       : 1100
# BLEU-4 Score        : 6.0441
# SacreBLEU Details   : BLEU = 6.04 13.8/6.9/4.5/3.1 (BP = 1.000 ratio = 2.792 hyp_len = 39923 ref_len = 14297)
# ==============================================


# ==================================================
#          📋 개별 파일 코퍼스 BLEU 평가 결과 📋
# ==================================================
# 파일명                            | BLEU-4 Score    | Samples
# --------------------------------------------------
# rag_test_results_1129_1.json   | 6.3479          | 100
# rag_test_results_1129_10.json  | 6.4141          | 100
# rag_test_results_1129_11.json  | 5.4387          | 100
# rag_test_results_1129_2.json   | 4.2161          | 100
# rag_test_results_1129_3.json   | 4.9771          | 100
# rag_test_results_1129_4.json   | 0.0638          | 100
# rag_test_results_1129_5.json   | 22.5835         | 100
# rag_test_results_1129_6.json   | 19.2541         | 100
# rag_test_results_1129_7.json   | 4.9970          | 100
# rag_test_results_1129_8.json   | 3.5341          | 100
# rag_test_results_1129_9.json   | 3.7315          | 100
# ==================================================