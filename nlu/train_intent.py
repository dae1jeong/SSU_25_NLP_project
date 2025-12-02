# ==============================================================================
# SSU_25_NLP_project - nlu/train_intent.py
#
# [개요]
# 사용자의 질문 의도를 분류하는 AI 모델을 훈련하는 스크립트입니다.
#
# [주요 역할]
# 1. 모델 기반: Hugging Face의 'klue/bert-base' 모델을 사용합니다.
# 2. 전처리: 텍스트와 라벨(의도)을 분리하고, LabelEncoder를 사용해 의도를 숫자로 변환합니다.
# 3. 모델 훈련: AutoModelForSequenceClassification을 사용하여 텍스트 분류 방식으로 모델을 Fine-tuning합니다.
# 4. 결과 저장: 훈련된 모델과 라벨 맵을 'models/intent_classifier' 경로에 저장하여 추론 시 사용합니다.
#
# [결과 활용]
# - RAG 파이프라인에서 검색 결과를 필터링하는 'intent' 값으로 활용됩니다.
# ==============================================================================
# 작업자 : 박대정

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nlu_utils import load_jsonl  # 우리가 만든 nlu_utils.py 임포트

# --- 1. 설정 ---
BASE_MODEL = "klue/bert-base"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "intent_train.jsonl")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "intent_classifier")

# --- 2. 데이터 전처리 ---
print("1. 데이터 로딩 및 전처리 시작...")

# 데이터 로드
try:
    raw_data = load_jsonl(DATA_FILE)
    if not raw_data:
        print(f"[오류] {DATA_FILE}에서 데이터를 불러오지 못했습니다.")
        exit()
        
    texts = [item['text'] for item in raw_data]
    labels = [item['label'] for item in raw_data]
except Exception as e:
    print(f"[오류] 데이터 로드 중 문제 발생: {e}")
    exit()

# 라벨(의도)을 숫자로 변환 (예: "잡담" -> 0, "강의평_조회" -> 1)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 라벨(숫자)과 의도(텍스트) 매핑 정보 저장 (나중에 추론 시 필요)
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
with open(os.path.join(MODEL_SAVE_PATH, "label_map.json"), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f" -> 라벨 맵 저장됨: {label_map}")

# 훈련/검증 데이터 분리 (80% 훈련, 20% 검증)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42
)

# --- 3. 토크나이저 및 모델 로딩 ---
print("2. 토크나이저 및 모델 로딩...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, 
    num_labels=len(label_map)  # 분류할 의도(라벨)의 개수
)

# 데이터셋 토큰화
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# --- 4. 훈련을 위한 데이터셋 클래스 정의 ---
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # PyTorch가 이해할 수 있는 딕셔너리 형태로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# --- 5. 훈련 ---
print("3. 모델 훈련 시작...")

# 훈련 설정
training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_SAVE_PATH, "checkpoints"), # 훈련 중 체크포인트 저장 경로
    num_train_epochs=3,          # 훈련 에포크 수 (데이터가 적으므로 3~5회)
    per_device_train_batch_size=2,  # 훈련 배치 크기
    per_device_eval_batch_size=2,   # 평가 배치 크기
    warmup_steps=10,             # 웜업 스텝
    weight_decay=0.01,           # 가중치 감쇠
    logging_dir='./logs',        # 로그 저장 경로
    logging_steps=1,
    
    # --- 아래는 모두 구버전에서 오류를 일으킬 수 있으니 일단 제거 ---
    # evaluation_strategy="steps", 
    # save_strategy="steps",       
    # load_best_model_at_end=True, 
    # eval_steps=1,                
    # save_steps=1,                
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# --- 6. 훈련된 모델 저장 ---
print("4. 훈련 완료 및 최종 모델 저장...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH) # 토크나이저도 함께 저장

print(f"[성공] 훈련된 의도 분류 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")