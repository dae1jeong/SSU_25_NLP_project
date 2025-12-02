
# ==============================================================================
# SSU_25_NLP_project - data.py 
#
# [개요]
# 이 스크립트는 챗봇의 '장기 기억'이 될 원본 데이터베이스를 구축하는
# ETL(Extract, Transform, Load) 파이프라인의 핵심 코드입니다.
# 흩어져 있는 JSONL 데이터 파일들을 읽어와 정제한 뒤, 하나의 SQLite DB로 통합합니다.
#
# [처리 대상 데이터]
# 1. 강의평 (Lecture Reviews):
#    - 파일: everytime_crawling/et_reviews_최종.jsonl
#    - 내용: 과목명, 교수명, 별점, 수강 학기, 강의평 텍스트
# 2. 공지사항 (Notices):
#    - 파일: notice crawling/image_captioned/ssu_rag_data_2025_v3.jsonl
#    - 내용: 공지 제목, 카테고리, 학과, 날짜, 본문(이미지 OCR 포함 Markdown)
# 3. 동아리 (Clubs):
#    - 파일: everytime_crawling/everytime_club_parsed.jsonl
#    - 내용: 동아리명(title), 소개글(all_text), 원본 링크
#


import sqlite3
import json
import os
import pandas as pd

# 현재 파일 기준으로 프로젝트 루트 경로 구하기
# .../SSU_25_NLP_project/data/data.py -> PROJECT_ROOT = .../SSU_25_NLP_project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 1단계: 데이터베이스 초기 설정 ---
def init_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 강의평
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
    
    # 2. 공지사항
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            category TEXT,
            post_date DATE,
            status TEXT,
            full_body_text TEXT,
            link TEXT UNIQUE,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. 동아리
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clubs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            club_name TEXT,
            category TEXT,
            description TEXT,
            recruitment_info TEXT,
            source_url TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f" -> 데이터베이스 '{db_path}' 및 3개 테이블 준비 완료")

# --- 2단계: 데이터 로딩 함수 ---
def load_data_from_jsonl(file_path):
    raw_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        print(f" -> [OK] JSONL 로딩 완료: {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f" -> [에러] JSONL 파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f" -> [에러] JSONL 파일 분석 중 오류: {e}")
        return None

# --- 3단계: 데이터 정제(Processing) 함수 ---

def process_reviews(raw_data):
    processed_list = []
    for review in raw_data:
        try:
            cleaned_star = float(review.get('review_star', 0))
            cleaned_semester = str(review.get('review_semester', '')).replace(' 수강자', '').strip()
            cleaned_text = str(review.get('review_text', '')).strip()

            professor_name = review.get('professor')
            if not professor_name or pd.isna(professor_name):
                professor_name = "정보 없음"
            else:
                professor_name = str(professor_name).strip()

            if cleaned_text:
                processed_list.append({
                    'subject_name': str(review.get('course_name', '')).strip(),
                    'professor_name': professor_name,
                    'star_rating': cleaned_star,
                    'semester': cleaned_semester,
                    'review_text': cleaned_text
                })
        except Exception:
            # 개별 레코드 에러는 무시
            pass
    return processed_list

def process_notices(raw_data):
    processed_list = []
    for item in raw_data:
        try:
            status_value = item.get('status') or '정보 없음'
            department_value = item.get('department') or '정보 없음'

            processed_list.append({
                'title': str(item.get('post_title', '')).strip(),
                'category': str(item.get('category', '')),
                'post_date': str(item.get('posted_date', '')),
                'status': str(status_value),
                'full_body_text': str(item.get('cleaned_markdown', '')).strip(),
                'link': str(item.get('source_url', '')),
                'department': str(department_value)
            })
        except Exception:
            pass
    return processed_list

def process_clubs(raw_data):
    """ 
    (수정됨 v5.3) 동아리 데이터 정제 
    - 값이 None일 경우 안전하게 처리 (.strip() 오류 방지)
    - title이 없으면 '제목 없음'으로 처리
    """
    processed_list = []
    for item in raw_data:
        try:
            # 1. 데이터 추출 
            raw_title = item.get('title')
            club_name = str(raw_title).strip() if raw_title else "제목 없음"
            
            raw_desc = item.get('all_text')
            description = str(raw_desc).strip() if raw_desc else ""
            
            raw_url = item.get('url')
            url = str(raw_url).strip() if raw_url else ""
            
            unique_id = str(item.get('id', ''))
            
            # 2. URL 생성
            if url and unique_id:
                unique_url = f"{url}#{unique_id}"
            elif unique_id:
                unique_url = f"club_id_{unique_id}"
            else:
                unique_url = f"club_{hash(description)}"

            category = '동아리' 

            if description: 
                processed_list.append({
                    'club_name': club_name,
                    'category': category,
                    'description': description,
                    'recruitment_info': '', 
                    'source_url': unique_url
                })
        except Exception as e:
            # 오류가 나도 멈추지 않고 출력만 함
            print(f"동아리 처리 중 오류 (무시됨): {e}")
            pass
            
    print(f" -> [OK] 동아리 {len(processed_list)}건 처리 완료")
    return processed_list

#  4단계: 데이터베이스에 저장하는 함수 
def save_data_to_db(db_path, processed_reviews, processed_notices, processed_clubs):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. 강의평 저장
        count_reviews = 0
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
            if cursor.rowcount > 0:
                count_reviews += 1
            
        # 2. 공지사항 저장
        count_notices = 0
        for notice in processed_notices:
            cursor.execute("""
                INSERT OR IGNORE INTO notices 
                (title, category, post_date, status, full_body_text, link, department) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                notice['title'],
                notice['category'],
                notice['post_date'],
                notice['status'],
                notice['full_body_text'],
                notice['link'],
                notice['department']
            ))
            if cursor.rowcount > 0:
                count_notices += 1

        # 3. 동아리 저장
        count_clubs = 0
        for club in processed_clubs:
            cursor.execute("""
                INSERT OR IGNORE INTO clubs 
                (club_name, category, description, recruitment_info, source_url) 
                VALUES (?, ?, ?, ?, ?)
            """, (
                club['club_name'],
                club['category'],
                club['description'],
                club['recruitment_info'],
                club['source_url']
            ))
            if cursor.rowcount > 0:
                count_clubs += 1
        
        conn.commit()
        print(f" -> [저장 완료] 강의평: {count_reviews}건, 공지: {count_notices}건, 동아리: {count_clubs}건")
        print(" -> [OK] 모든 데이터를 데이터베이스에 저장 완료")
        
    except Exception as e:
        conn.rollback()
        print(f" -> [DB 오류] 데이터 저장 중 문제 발생: {e}")
    finally:
        conn.close()

# --- 5단계: 전체 파이프라인 실행 ---
def main():
    # DB는 프로젝트 루트에 생성
    db_path = os.path.join(PROJECT_ROOT, "ssu_chatbot_data.db")
    print("1. 데이터베이스 초기화 시작...")
    init_database(db_path)

    # JSONL 경로는 프로젝트 루트 기준
    BASE_DIR = PROJECT_ROOT
    print(f" -> PROJECT_ROOT: {PROJECT_ROOT}")
    
    # 1. 강의평 처리
    print("\n2. 강의평 데이터 처리 시작...")
    review_file = os.path.join(BASE_DIR, "everytime_crawling", "et_reviews_최종.jsonl")
    raw_reviews = load_data_from_jsonl(review_file)
    reviews_list = process_reviews(raw_reviews) if raw_reviews else []
    
    # 2. 공지사항 데이터 처리
    print("\n3. 공지사항 데이터 처리 시작...")
    notice_file = os.path.join(BASE_DIR, "notice crawling", "image_captioned", "ssu_rag_data_2025_v3.jsonl")
    raw_notices = load_data_from_jsonl(notice_file)
    notices_list = process_notices(raw_notices) if raw_notices else []

    # 3. 동아리 처리
    print("\n4. 동아리 데이터 처리 시작...")
    club_file = os.path.join(BASE_DIR, "everytime_crawling", "everytime_club_parsed.jsonl") 
    raw_clubs = load_data_from_jsonl(club_file)
    clubs_list = process_clubs(raw_clubs) if raw_clubs else []
    
    # 4. 최종 저장
    print("\n5. 데이터베이스에 최종 저장...")
    save_data_to_db(db_path, reviews_list, notices_list, clubs_list)
    
    print("\n[모든 작업 완료]")

if __name__ == "__main__":
    main()