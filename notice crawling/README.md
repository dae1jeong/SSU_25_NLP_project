### 숭실대 공지사항 크롤링


게시글 관련 정보 크롤링: SoongSil_University_Notice ~.ipynb

결과: ssu_notices_2024_onwards.csv

<img width="1384" height="314" alt="스크린샷 2025-11-04 191940" src="https://github.com/user-attachments/assets/e1521ca9-4367-4d86-829a-eea77e45cd23" />


게시글 관련 정보 + 게시글의 세부사항까지 크롤링: notice_crawling.ipynb

결과: ssu_rag_data_2025.jsonl

실제 데이터 print
<img width="1008" height="473" alt="image" src="https://github.com/user-attachments/assets/ca74f62d-2939-4878-aa85-c49d65fbcb1a" />

포맷팅 적용 후 print
<img width="941" height="622" alt="image" src="https://github.com/user-attachments/assets/837721af-0756-4e2a-905c-4d16d5fea5ff" />

앞으로 RAG DB 구축을 위해서 필요한 일
- 이미지 캡셔닝
- 텍스트 청크 생성


- 벡터 임베딩 및 데이터베이스 구축
- Vector Database (ChromaDB, Pinecone, Weaviate)
- Vector: 청크 텍스트를 변환한 벡터 (검색 키)
- Context: 원본 청크 텍스트
- Metadata: source_url, post_title, status
