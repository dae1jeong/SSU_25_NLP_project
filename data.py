import sqlite3
import re
import csv
import json   # .jsonl 파일을 읽기 위해 import
import os     # 파일 경로를 안전하게 다루기 위해 import
from datetime import datetime

# --- 1단계: 데이터베이스 초기 설정 ---
def init_database(db_name="ssu_chatbot_data.db"):
    """
    데이터베이스 파일을 생성하고,
    [lecture_reviews] 테이블과 [notices] 테이블을 생성합니다.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # (기존) 강의평 테이블 생성
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
    
    # (신규) 공지사항 테이블 생성
    # (jsonl 파일의 source_url을 UNIQUE 키로 사용)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            category TEXT,
            post_date DATE,
            status TEXT,
            full_body_text TEXT,
            link TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f" -> 데이터베이스 '{db_name}' 및 2개 테이블 준비 완료")

# --- 2단계: 데이터 로딩 함수 ---

def load_review_data_from_csv(file_path):
    """
    (강의평) CSV 파일에서 강의평 데이터를 읽어옵니다.
    """
    raw_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append(row)
        print(f" -> [OK] 강의평 CSV 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] 강의평 CSV 파일을 찾을 수 없습니다: {file_path}")
        return None

def load_notice_data_from_jsonl(file_path):
    """
    (공지사항) .jsonl 파일에서 공지사항 데이터를 읽어옵니다.
    .jsonl 파일은 한 줄에 하나의 JSON 객체가 있습니다.
    """
    raw_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # 빈 줄이 아니면
                    raw_data.append(json.loads(line))
        print(f" -> [OK] 공지사항 JSONL 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] 공지사항 JSONL 파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f" -> [에러] JSONL 파일 분석 중 오류: {e}")
        return None

# --- 3단계: 데이터 정제(Processing) 함수 ---

def process_reviews(raw_data, subject, professor):
    """
    (강의평) 원본 데이터를 받아와 가공합니다.
    """
    processed_list = []
    for review in raw_data:
        try:
            cleaned_star = float(review['star'])
            cleaned_semester = review['semester'].replace(' 수강자', '').strip()
            cleaned_text = review['text'].strip()

            if cleaned_text:
                processed_list.append({
                    'subject_name': subject,
                    'professor_name': professor,
                    'star_rating': cleaned_star,
                    'semester': cleaned_semester,
                    'review_text': cleaned_text
                })
        except Exception as e:
            print(f" -> [처리 오류] 강의평 데이터: {e}, {review}")
            
    print(f" -> [OK] 강의평 {len(processed_list)}건 처리 완료")
    return processed_list

def process_notices(raw_data):
    """
    (공지사항) .jsonl 원본 데이터를 받아와 가공합니다.
    """
    processed_list = []
    for item in raw_data:
        try:
            processed_list.append({
                'title': item['post_title'].strip(),
                'category': item['category'],
                'post_date': item['posted_date'],
                'status': item['status'],
                'full_body_text': item['full_body_text'].strip(),
                'link': item['source_url']
            })
        except Exception as e:
            print(f" -> [처리 오류] 공지사항 데이터: {e}, {item}")
    
    print(f" -> [OK] 공지사항 {len(processed_list)}건 처리 완료")
    return processed_list

# --- 4단계: 데이터베이스에 저장하는 함수 ---
def save_data_to_db(db_name, processed_reviews, processed_notices):
    """
    정제된 강의평과 공지사항 데이터를 각각의 테이블에 저장합니다.
    (UNIQUE 제약조건으로 중복 데이터는 무시됩니다)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        # (기존) 강의평 저장
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
            
        # (신규) 공지사항 저장
        for notice in processed_notices:
            cursor.execute("""
                INSERT OR IGNORE INTO notices 
                (title, category, post_date, status, full_body_text, link) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                notice['title'],
                notice['category'],
                notice['post_date'],
                notice['status'],
                notice['full_body_text'],
                notice['link']
            ))
        
        conn.commit()
        print(" -> [OK] 모든 데이터를 데이터베이스에 저장 완료")
        
    except Exception as e:
        conn.rollback()
        print(f" -> [DB 오류] 데이터 저장 중 문제 발생: {e}")
    finally:
        conn.close()

# --- 5단계: 전체 파이프라인 실행 ---
def main():
    """
    모든 데이터 로딩, 처리, 저장 단계를 순서대로 실행합니다.
    """
    db_path = "ssu_chatbot_data.db"
    
    # 1. DB 초기화
    print("1. 데이터베이스 초기화 시작...")
    init_database(db_path)

    # 2. 파일 경로 설정 (스크립트 파일의 현재 위치 기준)
    #    (터미널 실행 위치와 상관없이 항상 정확한 경로를 찾음)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    
    # --- 강의평 처리 (자동화된 방식) ---
    print("\n2. 강의평 데이터 처리 시작...")
    review_folder_name = "everytime_crawling"
    review_folder_path = os.path.join(BASE_DIR, review_folder_name)

    # 최종적으로 DB에 저장할 모든 강의평을 담을 '큰' 리스트
    all_processed_reviews_list = [] 

    try:
        # 1. 'everytime_crawling' 폴더 안의 모든 파일 목록을 가져옴
        all_files = os.listdir(review_folder_path)
    except FileNotFoundError:
        print(f" -> [에러] 강의평 폴더를 찾을 수 없습니다: {review_folder_path}")
        all_files = [] # 에러가 나도 멈추지 않고, 빈 리스트로 다음 단계 진행

    # 2. 모든 파일을 순회하며 CSV 파일만 골라서 처리
    for filename in all_files:
        # 파일이 "reviews_"로 시작하고 ".csv"로 끝나는지 확인
        if filename.startswith("reviews_") and filename.endswith(".csv"):
            
            print(f"\n   --- '{filename}' 처리 중 ---")
            review_file_path = os.path.join(review_folder_path, filename)
            
            # 3. 파일명에서 과목명, 교수명 동적 추출
            try:
                # "reviews_"와 ".csv" 부분을 제거하고 '_'로 분리
                parts = filename.replace("reviews_", "").replace(".csv", "").split('_')
                subject_name = parts[0]
                professor_name = parts[1]
                print(f" -> 과목: {subject_name}, 교수: {professor_name}")
            except Exception as e:
                print(f" -> [경고] 파일명 파싱 오류: {filename} (오류: {e}). '알 수 없음'으로 대체합니다.")
                subject_name, professor_name = "알 수 없음", "알 수 없음" # 실패 시 기본값
                
            # 4. 개별 파일 로딩
            raw_reviews = load_review_data_from_csv(review_file_path)
            
            if raw_reviews:
                # 5. 개별 파일 처리
                processed_reviews = process_reviews(raw_reviews, subject_name, professor_name)
                # 6. 처리된 결과를 '큰' 리스트에 추가 (extend 사용)
                all_processed_reviews_list.extend(processed_reviews)

    print(f"\n -> [OK] 총 {len(all_processed_reviews_list)}건의 강의평 데이터를 성공적으로 처리했습니다.")
    # --- 강의평 처리 끝 ---


    # --- 공지사항 처리 ---
    print("\n3. 공지사항 데이터 처리 시작...")
    notice_folder = "notice crawling"  # 공지사항 JSONL이 있는 폴더 (README.md 참고)
    notice_jsonl_file = "ssu_rag_data_2025.jsonl"
    notice_file_path = os.path.join(BASE_DIR, notice_folder, notice_jsonl_file)
    
    raw_notices = load_notice_data_from_jsonl(notice_file_path)
    processed_notices_list = []
    if raw_notices:
        processed_notices_list = process_notices(raw_notices)
    
    # --- 최종 저장 ---
    print("\n4. 데이터베이스에 최종 저장...")
    # (수정) all_processed_reviews_list를 전달
    save_data_to_db(db_path, all_processed_reviews_list, processed_notices_list)
    
    print("\n[모든 작업 완료]")

# 이 스크립트가 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    main()

# 이 스크립트가 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    main()
