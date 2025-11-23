import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# .env 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_answer(question, ground_truth, predicted):
    """GPT-4o를 이용한 LLM-as-a-Judge 평가"""
    
    prompt = f"""
    너는 공정하고 엄격한 채점관이야. 아래 정보를 바탕으로 AI 챗봇의 답변 품질을 1점에서 5점 사이로 평가해줘.
    
    [질문]: {question}
    [정답(기준)]: {ground_truth}
    [챗봇 답변]: {predicted}
    
    [평가 기준]
    5점: 정답의 핵심 내용을 빠짐없이 포함하며, 설명이 정확하고 자연스러움.
    4점: 핵심 내용은 포함했으나, 사소한 정보가 누락되거나 약간 부자연스러움.
    3점: 정답의 일부만 맞거나, 불필요한 정보가 섞여 있어 명확하지 않음.
    2점: 질문과 관련은 있으나 핵심 정보가 틀렸거나 엉뚱한 대답을 함.
    1점: 질문을 이해하지 못했거나 완전히 틀린 정보를 제공함.
    
    [출력 형식 (JSON)]:
    {{
        "score": 점수(1~5 정수),
        "reason": "점수를 부여한 이유 (한 문장으로)"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # ✅ 채점은 똑똑한 4o 사용!
            messages=[{"role": "system", "content": "너는 채점관이야. JSON으로만 답해."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"score": 0, "reason": "채점 중 에러 발생"}

def main():
    input_path = "Evaluation/rag_test_results.json"
    if not os.path.exists(input_path):
        print("❌ 실행 결과 파일이 없습니다. 2번 코드를 먼저 실행하세요.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    print(f"⚖️ GPT-4o 채점관이 {len(results)}개의 답안을 채점합니다...")
    
    total_score = 0
    evaluated_results = []
    
    for item in tqdm(results):
        eval_result = evaluate_answer(item['question'], item['ground_truth'], item['predicted_answer'])
        
        item['score'] = eval_result['score']
        item['reason'] = eval_result['reason']
        
        total_score += item['score']
        evaluated_results.append(item)
        
    if len(results) > 0:
        avg_score = total_score / len(results)
    else:
        avg_score = 0
        
    print(f"\n📊 [최종 성적표]")
    print(f"   - 총 문제 수: {len(results)}개")
    print(f"   - 평균 점수: {avg_score:.2f} / 5.0 점")
    
    # 결과 저장
    with open("Evaluation/data/final_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_results, f, ensure_ascii=False, indent=2)
        
    print("✅ 채점 완료! 상세 결과: Evaluation/data/final_evaluation_report.json")

if __name__ == "__main__":
    main()