# SSU_25_NLP_project
25년도 2학기 자연어처리(NLP) 프로젝트 

팀원 : 권은이, 박대정, 박원형, 박채은

안녕하세요ㅎㅎ
commit test



추가 사항

+) generate_dataset_chunking 추가 -> chunking 추가함.
<details>
<summary>자세한 내용 보기</summary>

기존 `generate_dataset.py`  
→ 청킹 없이 전체 문서를 context로 사용하며 합성 데이터셋 QA쌍 생성  

그러나 실제 궁금했슈는 청킹된 문서를 통해 답변을 생성함.  
따라서 합성 데이터셋 생성 시에도 청킹 (vector_db 생성 시와 동일한 방법) 해야 함.

======= 현재 임베딩 모델 =======  
- SOURCE_DB_PATH = "ssu_chatbot_data.db"  
- VECTOR_DB_PATH = "./chroma_db"  
- COLLECTION_NAME = "ssu_knowledge_base"  
- EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"  

- CHUNK_SIZE = 400     # 청크 글자 수  
- CHUNK_OVERLAP = 50   # 청크 겹침 글자 수  
- CHROMA_ADD_BATCH_SIZE = 5000  

**변경 후**  
궁금했슈와 동일하게, review 데이터는 원문을 청킹하지 않고, notice / club 데이터는 원문을 청킹하여 합성 데이터셋을 생성함.

</details>

+) 중간결과물 얻기 위해 chroma_db.py 수정
<details>
  <summary>자세한 내용 보기</summary>
  현재 합성데이터셋 만들 때 vector_db에서 청킹을 하는 방식, 사용한 db가 같음.
그러나 매번 청킹을 또 하는 것은 비효율적이니 
vector_db.py에서 청킹 + 임베딩을 한번에 하므로
다시 실행시켜서 청킹 시 중간 결과물을 저장해두고
chroma db를 다시 만들자.

그럼 generate_dataset 시 청킹을 하지 않아도 됨. ===> 수정 코드 만듦

</details>

+) rag_pipeline 수정함. ragas 이용할 수 있도록
<details>
  <summary>자세히보기</summary>
  평가 시 현재 코드 처럼 llm 평가도 유지하고, + ragas (정량적 평가 지표)도 추가하자.
ragas는 llm을 도구로 활용하여 rag를 자동화된 정량적 평가 지표로 측정함.
-> 현재 코드보다 좀 더 정밀한 평가.가 가능.

| **RAGAS 지표**                     | **평가 대상**           | **설명**                                                       |
| -------------------------------- | ------------------- | ------------------------------------------------------------ |
| **Faithfulness (충실도)**           | **생성 (Generation)** | 생성된 답변의 내용이 **제공된 컨텍스트**에서 뒷받침되는가? (환각(Hallucination) 방지)    |
| **Context Recall (컨텍스트 회수율)**    | **검색 (Retrieval)**  | **정답(Ground Truth)**을 도출하는 데 필요한 정보가 검색된 컨텍스트에 얼마나 포함되어 있는가? |
| **Answer Relevancy (답변 관련성)**    | **생성 (Generation)** | 생성된 답변이 **사용자의 질문**에 얼마나 직접적이고 적절하게 대응하는가?                   |
| **Context Relevancy (컨텍스트 관련성)** | **검색 (Retrieval)**  | 검색된 컨텍스트가 **질문**에 얼마나 집중하고 불필요한 정보가 없는가?                     |

</details>
+) ragas로 평가하는 코드 3_evaluate_ragas.py 추가...


앞으로
+) vector 실행해서 chunked_data.jsonl과 chorma db 획득\n
+) chroma db 사용해서 vector Search 하도록 rag_pipline 수정\n
+) 3_evaluate_ragas.py 디버깅\n
