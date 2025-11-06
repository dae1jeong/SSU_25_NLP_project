### 숭실대 공지사항 크롤링


#### 1 메타데이터 크롤링

메타데이터: 게시글 제목, 링크, 조회수, 업로드일, 업로드 부서 등.

게시글 관련 정보 크롤링: SoongSil_University_Notice ~.ipynb

** 결과: ssu_notices_2024_onwards.csv **

<img width="1384" height="314" alt="스크린샷 2025-11-04 191940" src="https://github.com/user-attachments/assets/e1521ca9-4367-4d86-829a-eea77e45cd23" />



#### 2 게시글 내용 크롤링

게시글 관련 정보 + 게시글의 세부사항까지 크롤링: notice_crawling.ipynb

** 결과: ssu_rag_data_2025.jsonl **

실제 데이터 print
<img width="1008" height="473" alt="image" src="https://github.com/user-attachments/assets/ca74f62d-2939-4878-aa85-c49d65fbcb1a" />

포맷팅 적용 후 print
<img width="941" height="622" alt="image" src="https://github.com/user-attachments/assets/837721af-0756-4e2a-905c-4d16d5fea5ff" />

앞으로 RAG DB 구축을 위해서 필요한 일
- 이미지 캡셔닝
- 텍스트 청크 생성

참고문서: https://bcho.tistory.com/1400
https://devocean.sk.com/blog/techBoardDetail.do?ID=167446&boardType=techBlog

- 벡터 임베딩 및 데이터베이스 구축
- Vector Database (ChromaDB, Pinecone, Weaviate)
- Vector: 청크 텍스트를 변환한 벡터 (검색 키)
- Context: 원본 청크 텍스트
- Metadata: source_url, post_title, status


#### 3 게시글 내용 정제

- 게시글 내용 재크롤링 (본문을 markdown 형식으로 저장하기 위해. html2markdown 라이브러리 사용)
** 결과: ssu_rag_data_2025_v2.json1 **
  
  게시글 본문의 속성명 변경: "full_body_text" -> "full_body_markdown"
  
  게시글 레코드: 1112개 (1112개 게시글)

- 게시글 본문에서 노이즈 제거
  게시글 본문에 있는 링크들을 새 속성으로 추출 (문맥 이해에 방해되기 때문)
  
  ** 결과: stage1_cleaned_data.jsonl **
  
  게시글 본문의 속성명 추가: "cleaned_markdown"


  (ex) `노이즈 제거 전: "full_body_markdown": [**슈패스 신청기간: 11/3(월) ~ 21(금) ‘저자강연회’ 검색 후 신청*선착순마감**](https://path.ssu.ac.kr/)  ![](https://oasis.ssu.ac.kr/pyxis-api/attachments/BULLETIN/b076854e-4f68-44db-b559-6b05f0ea3a7f)] 안녕하세요. 중앙도서관입니다.\n\n ...`
  
       노이즈 제거 후:  "cleaned_markdown": "안녕하세요. 중앙도서관입니다.\n\n ... ",
                        "extra_links": [
                            {
                                "text": "슈패스 신청기간: 11/3(월) ~ 21(금) ‘저자강연회’ 검색 후 신청선착순마감",
                                "href": "https://path.ssu.ac.kr/"
                            }
                        ]

  
- 이미지 캡셔닝 완료
  게시글 본문이 없고 이미지만 있는 공지사항이 존재.
  
  노이즈 제거된 게시글 본문의 글자수 < 50 이면 이미지 캡셔닝 진행.
  
  gemini API를 통해 gemini flash를 불러 해당 게시글의 모든 이미지 url과 다음 프롬프트를 주어 이미지 캡셔닝함.
  
  생성된 캡셔닝은 레코드의 cleaned_markdown 속성에 저장됨.

  `prompt_text = (
                "이 문서를 상세히 분석하여 표, 일정, 날짜 등 모든 정보를 OCR을 통해 추출하고 "
                "키:값 형태로 구조화해 주세요. 이 이미지는 학교 공지사항 포스터입니다."
            )`

 ** 결과: ssu_rag_2025_v3.jsonl **

 (ex) `이미지 캡셔닝 전: "cleaned_markdown": ""`
 
      이미지 캡셔닝 후: `"cleaned_markdown": "### 이미지 분석 결과 ###
                        다음은 이미지에서 추출된 정보이며, OCR을 통해 분석된 표, 일정, 날짜 등의 모든 데이터를 키:값 형태로 구조화한 것입니다.
                        
                        **공지사항 제목:** (재)전기공사공제조합장학회 2026년도 제21기 장학생 선발 공고
                        **공지 내용:** 전기산업의 미래를 이끌어갈 인재를 지원하기 위해 전기공사공제조합장학회가 장학생을 선발합니다.
                        **장학금 지원 성격:** 등록금 실비 지원 (등록금성장학금)
                        
                        --- ..."`

 
  



