# ==============================================================================
# SSU_25_NLP_project - nlu/train_slot.py
#
# [개요]
# 사용자의 질문에서 핵심 개체명(Slot/NER)을 추출하는 AI 모델을 훈련하는 스크립트입니다.
# (예: "김OO 교수님"에서 '김OO' 추출)
#
# [주요 역할]
# 1. 모델 기반: Hugging Face의 'klue/bert-base'를 사용합니다.
# 2. 라벨 정렬: 'tokenize_and_align_labels' 함수를 통해 단어 라벨을 BERT 토큰에 맞춰 정렬합니다.
#    - 이는 슬롯 추출에서 가장 복잡하고 중요한 전처리 과정입니다.
# 3. 모델 훈련: AutoModelForTokenClassification을 사용하여 토큰 분류 방식으로 Fine-tuning합니다.
# 4. 결과 저장: 훈련된 모델과 라벨 맵을 'models/slot_extractor' 경로에 저장하여 추론 시 사용합니다.
#
# [결과 활용]
# - RAG 파이프라인에서 검색 결과를 좁히는 필터(교수명, 학과명)로 활용됩니다.
# ==============================================================================

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
from nlu_utils import load_jsonl # 우리가 만든 nlu_utils.py 임포트

# --- 1. 설정 ---
BASE_MODEL = "klue/bert-base"  # (오류 수정) klue/bert-base 사용
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "slot_train.jsonl")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "slot_extractor")

# --- 2. 데이터 로딩 및 라벨링 ---
print("1. 데이터 로딩 및 전처리 시작...")

try:
    raw_data = load_jsonl(DATA_FILE)
    if not raw_data:
        print(f"[오류] {DATA_FILE}에서 데이터를 불러오지 못했습니다.")
        exit()

    # 'tokens' (단어 리스트)와 'tags' (태그 리스트)를 분리
    texts = [item['tokens'] for item in raw_data]
    labels = [item['tags'] for item in raw_data]

except Exception as e:
    print(f"[오류] 데이터 로드 중 문제 발생: {e}")
    exit()

# 모든 고유한 태그(라벨)를 찾아서 숫자와 매핑
unique_tags = set(tag for tag_list in labels for tag in tag_list)
label_map = {tag: i for i, tag in enumerate(unique_tags)}
id_to_label_map = {i: tag for tag, i in label_map.items()}

# 라벨 맵 저장 (나중에 추론 시 필요)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
with open(os.path.join(MODEL_SAVE_PATH, "label_map.json"), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f" -> 라벨 맵 저장됨: ({len(label_map)}개) {label_map.keys()}")

# 훈련/검증 데이터 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 3. 토크나이저 및 모델 로딩 ---
print("2. 토크나이저 및 모델 로딩...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# AutoModelForTokenClassification 사용
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL, 
    num_labels=len(label_map),     # 분류할 태그(라벨)의 개수
    id2label=id_to_label_map,      # 숫자 -> 태그명 매핑
    label2id=label_map             # 태그명 -> 숫자 매핑
)

# --- 4. 토큰화 및 라벨 정렬 (Label Alignment) ---
# 슬롯 추출에서 가장 복잡한 부분입니다.
# (예: 'AI 입문' 2단어 -> 토큰화 -> 'ai', '입', '##문' 3토큰)
# 'B-과목명', 'I-과목명' 태그를 토큰에 맞게 정렬해야 합니다.

def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(
        texts, 
        truncation=True, 
        padding=False, # DataCollator가 패딩을 처리
        max_length=128, 
        is_split_into_words=True # 텍스트가 이미 단어 리스트임을 알림
    )
    
    aligned_labels = []
    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP] 같은 스페셜 토큰은 -100 (loss 계산에서 무시됨)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 단어의 첫 번째 토큰
                label_ids.append(label_map[label_list[word_idx]])
            else:
                # 단어의 두 번째 이상 토큰 (Sub-token)
                # 동일하게 -100으로 설정하여, 단어의 첫 토큰만 훈련
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        aligned_labels.append(label_ids)
        
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

train_encodings = tokenize_and_align_labels(train_texts, train_labels)
val_encodings = tokenize_and_align_labels(val_texts, val_labels)

# --- 5. 훈련을 위한 데이터셋 클래스 정의 ---
class SlotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # input_ids, token_type_ids, attention_mask, labels
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["labels"])

train_dataset = SlotDataset(train_encodings)
val_dataset = SlotDataset(val_encodings)

# Data Collator: 배치(batch)를 만들 때, 
# 각 문장 길이에 맞게 동적으로 패딩(padding)을 추가해줌
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 6. 훈련 ---
print("3. 모델 훈련 시작...")

# 훈련 설정 (train_intent.py에서 성공했던 '안전한' 버전)
training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_SAVE_PATH, "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 데이터가 적으므로 2로 설정
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1,
    
    # --- 버전 오류를 피하기 위해 평가/저장 관련 인수는 모두 제거 ---
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
    eval_dataset=val_dataset,       # 평가는 하지만, 최고 모델 저장은 X
    data_collator=data_collator,    # 동적 패딩 적용
)

trainer.train()

# --- 7. 훈련된 모델 저장 ---
print("4. 훈련 완료 및 최종 모델 저장...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"[성공] 훈련된 슬롯 추출 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")