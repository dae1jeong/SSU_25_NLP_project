최종본 ssu_rag_data_2025_v3.jsonl

레코드 개수: 1112

컬럼 종류 및 설명:
source_url, post_title, category, department, views, posted_date, full_body_markdown, cleaned_markdown, image_urls, download_links,  extra_links, captined


source_url: 게시글의 url

post_title: 게시글의 제목

category: 게시글의 카테고리

department: 게시글을 올린 부서

views: 조회수 (크롤링 당시)

posted_date: 게시글을 올린 날짜

full_body_markdown: 정제 전 게시글 본문 -> 사실상 필요 없음. cleaned_markdown으로 대체.

**cleaned_markdown: 게시글 본문 (정제 후)**

image_urls: 게시글에 첨부된 이미지 url

download_links: 게시글에 첨부된 문서 다운로드 링크 (ex. 장학금 게시글에 첨부된 장학금 지원서 링크)

extra_links: 기타 게시글에 첨부된 링크

captined: gemini flash 2.5에 의한 이미지 캡셔닝 여부 (ex. true, false)
