#원본데이터를 db에 저장하는 코드
#크롤링 폴더에서 원본 파일 로드 후 불필요한 텍스트 제거, 공백 정리 등 전처리 수행
#정제된 데이터를 SQLITE DB에 저장


import sqlite3
import re
import csv
import json   
import os    
from datetime import datetime
import pandas as pd


    

#데이터베이스 초기 설정
def init_database(db_name="ssu_chatbot_data.db"):
    """
    데이터베이스 파일을 생성하고,
    [lecture_reviews] 테이블과 [notices] 테이블을 생성합니다.
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
    
    # 공지사항 테이블 생성
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

#  데이터 로딩 함수

def load_data_from_csv(file_path):
    """ (범용) CSV 파일에서 데이터를 읽어옵니다. """
    raw_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append(row)
        print(f" -> [OK] CSV 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] CSV 파일을 찾을 수 없습니다: {file_path}")
        return None

def load_notice_data_from_jsonl(file_path):
    """ (신규) .jsonl 파일에서 공지사항 데이터를 읽어옵니다. """
    raw_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    raw_data.append(json.loads(line))
        print(f" -> [OK] 공지사항 JSONL 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] 공지사항 JSONL 파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f" -> [에러] JSONL 파일 분석 중 오류: {e}")
        return None

#  데이터 정제 함수 

def process_reviews(raw_data):
    """
    (강의평 수정) 
    'et_reviews_nonzero.csv'의 실제 컬럼명에 맞게 수정합니다.
    """
    processed_list = []
    for review in raw_data:
        try:
            cleaned_star = float(review['review_star'])
            cleaned_semester = review['review_semester'].replace(' 수강자', '').strip()
            cleaned_text = review['review_text'].strip()

            # professor 값이 비어있는 경우를 대비해 None으로 치환
            professor_name = review.get('professor')
            if not professor_name or pd.isna(professor_name):
                professor_name = "정보 없음"
            else:
                professor_name = professor_name.strip()

            if cleaned_text:
                processed_list.append({
                    'subject_name': review['course_name'].strip(),
                    'professor_name': professor_name,
                    'star_rating': cleaned_star,
                    'semester': cleaned_semester,
                    'review_text': cleaned_text
                })
        except KeyError as e:
            print(f" -> [처리 오류] 강의평 컬럼 오류: {e}. (해당 row 무시)")
        except Exception as e:
            print(f" -> [처리 오류] 강의평 데이터: {e}, {review} (해당 row 무시)")
            
    print(f" -> [OK] 강의평 {len(processed_list)}건 처리 완료")
    return processed_list

def process_notices(raw_data):
    """
    (공지사항 수정) 
    v3.jsonl 원본 데이터를 가공하고, 'status' 키 오류를 수정합니다.
    """
    processed_list = []
    for item in raw_data:
        try:
           
            
            status_value = item.get('status', '정보 없음') 

            processed_list.append({
                'title': item['post_title'].strip(),
                'category': item['category'],
                'post_date': item['posted_date'],
                'status': status_value, 
                'full_body_text': item['cleaned_markdown'].strip(),
                'link': item['source_url']
            })
        except KeyError as e:
            
            print(f" -> [처리 오류] 공지사항 JSONL 키(key) 불일치: {e}. (해당 item 무시)")
        except Exception as e:
            print(f" -> [처리 오류] 공지사항 데이터: {e}, {item} (해당 item 무시)")
    
    print(f" -> [OK] 공지사항 {len(processed_list)}건 처리 완료")
    return processed_list

#  데이터베이스에 저장하는 함수 
def save_data_to_db(db_name, processed_reviews, processed_notices):
    """
    정제된 강의평과 공지사항 데이터를 각각의 테이블에 저장합니다.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        #  강의평 저장
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
            
        #  공지사항 저장
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

# 전체 파이프라인 실행
def main():
    """
    모든 데이터 로딩, 처리, 저장 단계를 순서대로 실행합니다.
    """
    db_path = "ssu_chatbot_data.db"
    
    print("1. 데이터베이스 초기화 시작...")
    init_database(db_path)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    
    #  강의평 처리 
    print("\n2. 강의평 데이터 처리 시작...")
    review_folder_name = "everytime_crawling"
    review_csv_file = "et_reviews_nonzero.csv" 
    review_file_path = os.path.join(BASE_DIR, review_folder_name, review_csv_file)
    
    raw_reviews = load_data_from_csv(review_file_path)
    
    all_processed_reviews_list = []
    if raw_reviews:
        all_processed_reviews_list = process_reviews(raw_reviews)
    
    print(f"\n -> [OK] 총 {len(all_processed_reviews_list)}건의 강의평 데이터를 성공적으로 처리했습니다.")
    


    #  공지사항 처리 
    print("\n3. 공지사항 데이터 처리 시작...")
    
    
    notice_folder = "notice crawling"
    notice_subfolder = "image_captioned"
    notice_jsonl_file = "ssu_rag_data_2025_v3.jsonl" 
    
    
    notice_file_path = os.path.join(BASE_DIR, notice_folder, notice_subfolder, notice_jsonl_file)
    
    raw_notices = load_notice_data_from_jsonl(notice_file_path) 
    processed_notices_list = []
    if raw_notices:
        processed_notices_list = process_notices(raw_notices)
    
    # 최종 저장 
    print("\n4. 데이터베이스에 최종 저장...")
    save_data_to_db(db_path, all_processed_reviews_list, processed_notices_list)
    
    print("\n[모든 작업 완료]")


if __name__ == "__main__":
    main()