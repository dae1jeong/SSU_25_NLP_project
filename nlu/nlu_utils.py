import json
import os

def load_jsonl(file_path):
    """
    .jsonl 파일을 읽어서 리스트에 담아 반환합니다.
    
    Args:
        file_path (str): 읽어올 .jsonl 파일의 경로
        
    Returns:
        list: 각 줄의 JSON 객체가 담긴 리스트
    """
    data = []
    
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
        return data
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 빈 줄이 아닌 경우에만 파싱
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"[성공] 데이터 로드 완료: {file_path} ({len(data)}건)")
        return data
        
    except json.JSONDecodeError as e:
        print(f"[오류] JSONL 파일 파싱 중 오류 발생: {e}")
        print(f" -> 문제가 된 줄: {line}")
        return []
    except Exception as e:
        print(f"[오류] 파일을 읽는 중 알 수 없는 오류 발생: {e}")
        return []

# --- 사용 예시 ---
# (이 스크립트를 직접 실행할 때만 아래 코드가 동작합니다)
if __name__ == "__main__":
    
    # 이 파일(nlu_utils.py)이 있는 nlu 폴더를 기준으로 data 폴더 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. 의도 분류 데이터 로드 테스트
    print("--- 의도 분류 데이터 테스트 ---")
    intent_file = os.path.join(data_dir, "intent_train.jsonl")
    intent_data = load_jsonl(intent_file)
    if intent_data:
        print(f" -> 첫 번째 데이터: {intent_data[0]}")

    print("\n" + "="*30 + "\n")

    # 2. 슬롯 추출 데이터 로드 테스트
    print("--- 슬롯 추출 데이터 테스트 ---")
    slot_file = os.path.join(data_dir, "slot_train.jsonl")
    slot_data = load_jsonl(slot_file)
    if slot_data:
        print(f" -> 첫 번째 데이터: {slot_data[0]}")