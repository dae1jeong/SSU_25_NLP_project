import sqlite3
import re
import csv
from datetime import datetime
import os

# --- 1단계: 데이터베이스 초기 설정 ---
def init_database(db_name="ssu_chatbot_data.db"):
    """
    데이터베이스 파일을 생성하고, [lecture_reviews] 테이블이 없으면 새로 생성합니다.
    (공지사항, 학사일정 테이블은 제외)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 강의평 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lecture_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT,
            professor_name TEXT,
            star_rating REAL,
            semester TEXT,
            review_text TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f" -> 데이터베이스 '{db_name}' 및 'lecture_reviews' 테이블 준비 완료")

# --- 2단계: CSV 데이터 로딩 함수 ---
def load_review_data_from_csv(file_path):
    """
    (실제 데이터) CSV 파일에서 강의평 데이터를 읽어옵니다.
    """
    raw_data = []
    try:
        # 'utf-8-sig'는 한글 CSV 파일의 인코딩 깨짐을 방지합니다.
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            # csv.DictReader는 첫 줄을 헤더(key)로, 나머지를 value로 읽어옵니다.
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append(row)
        print(f" -> [REAL] CSV 파일 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] CSV 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f" -> [에러] CSV 파일 로딩 중 오류: {e}")
        return None

# --- 3단계: 데이터 정제(Processing) 함수 ---
def process_reviews(raw_data, subject, professor):
    """
    (실제) 강의평 원본 데이터를 받아와 가공합니다.
    """
    processed_list = []
    for review in raw_data:
        try:
            # 별점(star)을 문자열("5.00")에서 숫자(5.0)로 변환
            cleaned_star = float(review['star'])
            
            # 수강 학기 정보에서 불필요한 ' 수강자' 텍스트 제거
            cleaned_semester = review['semester'].replace(' 수강자', '').strip()
            
            # 리뷰 텍스트 양쪽 공백 제거
            cleaned_text = review['text'].strip()

            # 비어있는 리뷰는 저장하지 않음
            if cleaned_text:
                processed_list.append({
                    'subject_name': subject,
                    'professor_name': professor,
                    'star_rating': cleaned_star,
                    'semester': cleaned_semester,
                    'review_text': cleaned_text
                })
        except Exception as e:
            print(f" -> [처리 오류] 리뷰 데이터 처리 중 문제 발생: {e}, {review}")
            
    print(f" -> [REAL] 강의평 {len(processed_list)}건 처리 완료")
    return processed_list

# --- 4단계: 데이터베이스에 저장하는 함수 ---
def save_data_to_db(db_name, processed_reviews):
    """
    정제된 강의평 데이터를 데이터베이스에 저장합니다.
    (UNIQUE 제약조건으로 중복 데이터는 무시됩니다)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        # 강의평 저장
        for review in processed_reviews:
            cursor.execute("""
                INSERT OR IGNORE INTO lecture_reviews 
                (subject_name, professor_name, star_rating, semester, review_text) 
                VALUES (?, ?, ?, ?, ?)
            """, (
                review['subject_name'],
                review['professor_name'],
                review['star_rating'],
                review['semester'],
                review['review_text']
            ))
        
        conn.commit() # 모든 변경사항을 최종 저장
        print(" -> 처리된 데이터를 데이터베이스에 저장 완료")
        
    except Exception as e:
        conn.rollback() # 오류 발생 시 모든 변경사항 되돌리기
        print(f" -> [DB 오류] 데이터 저장 중 문제 발생: {e}")
    finally:
        conn.close() # 성공하든 실패하든 연결 종료

# --- 5단계: 전체 파이프라인 실행 ---
def main():
    """
    모든 단계를 순서대로 실행하는 메인 함수
    """
    db_path = "ssu_chatbot_data.db"
    
    print("1. 데이터베이스 초기화 시작...")
    init_database(db_path)

    # --- (Real) 강의평 데이터 처리 ---
    print("\n2. 실제 데이터(강의평) 처리 시작...")

    # ----- 이 부분을 수정합니다 -----
    
    # 1. data.py 파일의 현재 위치(폴더)를 가져옵니다.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    file_name = "reviews_시스템프로그래밍_이은지.csv" # CSV 파일 이름
    
    # 2. 폴더 경로와 파일 이름을 바로 합칩니다. (중간 폴더 제거)
    csv_file_name = os.path.join(BASE_DIR, file_name) # <-- folder_name 변수 제거
    
    print(f" -> 파일 찾기 시도 (수정된 경로): {csv_file_name}") # 경로가 올바른지 로그로 확인

    # 3. 파일 이름에서 과목명과 교수명 자동 추출
    try:
        parts = file_name.replace("reviews_", "").replace(".csv", "").split('_')
        subject_name = parts[0]
        professor_name = parts[1]
        print(f" -> 파일명 분석: 과목명={subject_name}, 교수명={professor_name}")
    except Exception:
        print(" -> [경고] 파일명 분석 실패. 기본값을 사용합니다.")
        subject_name = "시스템프로그래밍"
        professor_name = "이은지"
    
    # ----- 여기까지 수정 -----
        
    raw_reviews = load_review_data_from_csv(csv_file_name)
    
    processed_reviews_list = []
    if raw_reviews is not None:
        processed_reviews_list = process_reviews(raw_reviews, subject_name, professor_name)
    
    # --- 최종 저장 ---
    print("\n3. 데이터베이스에 최종 저장...")
    save_data_to_db(db_path, processed_reviews_list)
    
    print("\n[모든 작업 완료]")

# 이 스크립트가 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    main()