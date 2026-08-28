# 🔎 RAG Retrieval System

> **청년정책 데이터를 기반으로 검색 성능을 정량적으로 평가하고 개선하는 RAG 시스템 프로젝트**

## 📌 Project Overview

본 프로젝트는 **Retrieval-Augmented Generation(RAG)** 구조를 직접 설계하고 구현하며, 검색 단계의 성능을 정량적으로 평가하고 개선하는 것을 목표로 합니다.

단순한 질의응답 서비스 구현보다 **데이터 수집 → 전처리 → Chunking → Embedding → Vector DB → Retrieval → Evaluation**으로 이어지는 RAG 파이프라인을 구축하고, 설정 변화에 따른 검색 성능을 비교·분석하는 데 중점을 둡니다.

데이터는 **온통청년 청년정책 Open API**를 통해 수집합니다.

초기 Baseline 단계에서는 전체 정책 데이터에서 **카테고리별 일정 비율을 추출한 표본 데이터셋**을 활용하여 RAG 파이프라인과 평가 환경을 구축합니다.

이후 Baseline 실험을 통해 결정한 설정을 기반으로 **온통청년 전체 정책 데이터셋**으로 확장하여 실제 데이터 규모에서 검색 시스템을 구축하고 성능을 분석합니다.

---

## 🎯 Goals

* RAG 검색 파이프라인 직접 설계 및 구현
* 온통청년 Open API 기반 정책 데이터 수집 파이프라인 구축
* 공통 데이터 스키마 및 전처리 구조 설계
* 카테고리별 표본 데이터를 활용한 Baseline RAG 구축
* Chunking / Embedding / Retrieval 설정에 따른 검색 성능 비교
* Vector DB 기반 정책 문서 검색 시스템 구축
* 정량적 평가 지표를 활용한 Retrieval 성능 검증
* 실험 결과와 설정을 추적할 수 있는 재현 가능한 실험 환경 구축
* Baseline에서 검증한 구조를 전체 정책 데이터셋으로 확장

---

## 🏗️ RAG Pipeline

```text
온통청년 Open API
        │
        ▼
Raw Policy Data
        │
        ▼
Preprocessing
        │
        ▼
Common Document Schema
        │
        ├───────────────────────────┐
        │                           │
        ▼                           ▼
Baseline Dataset              Full Dataset
카테고리별 표본 추출             전체 정책 데이터
        │                           │
        ▼                           │
Chunking                         │
        │                           │
        ▼                           │
Embedding                        │
        │                           │
        ▼                           │
ChromaDB                         │
        │                           │
        ▼                           │
Retrieval                        │
        │                           │
        ▼                           │
Evaluation                       │
        │                           │
        ▼                           │
Baseline 설정 확정 ────────────────┘
                                    │
                                    ▼
                           Full Dataset RAG
```

---

## 🗂️ Dataset

### 온통청년 청년정책 데이터

프로젝트의 Baseline과 Main 단계 모두 **온통청년 청년정책 데이터**를 사용합니다.

온통청년 Open API를 통해 정책 데이터를 수집한 뒤 프로젝트에서 정의한 공통 데이터 형식으로 변환합니다.

### Baseline Dataset

전체 데이터를 바로 사용하기 전에 RAG 파이프라인과 평가 환경을 구축하고 다양한 설정을 빠르게 실험하기 위한 데이터셋입니다.

전체 정책 데이터에서 특정 카테고리에 데이터가 편중되지 않도록 **카테고리별 일정 비율을 추출하는 층화 표본 추출(Stratified Sampling)** 방식을 사용합니다.

```text
전체 온통청년 정책 데이터
          │
          ├── 일자리 ────── 일정 비율 추출
          ├── 주거 ──────── 일정 비율 추출
          ├── 교육 ──────── 일정 비율 추출
          ├── 복지·문화 ─── 일정 비율 추출
          └── 기타 ──────── 일정 비율 추출
                         │
                         ▼
                  Baseline Dataset
```

※ 실제 카테고리명과 추출 비율은 수집된 온통청년 데이터 및 프로젝트 설정을 기준으로 관리합니다.

**주요 목적**

* 데이터 파이프라인 검증
* Baseline RAG 구축
* Retrieval 평가 환경 구축
* Chunking 설정 비교
* Embedding Model 비교
* Top-k 설정 비교
* Hit@k / MRR 측정

### Main Dataset

Baseline 실험이 완료된 이후에는 **온통청년에서 수집한 전체 정책 데이터**를 사용합니다.

Baseline 단계에서 검증한 전처리, Chunking, Embedding 및 Retrieval 설정을 전체 데이터에 적용하여 데이터 규모가 증가한 환경에서 검색 성능을 분석합니다.

```text
Baseline
카테고리별 표본 데이터
        │
        ▼
RAG 구조 및 평가 환경 검증
        │
        ▼
설정 비교 실험
        │
        ▼
Baseline 설정 확정
        │
        ▼
Main
온통청년 전체 정책 데이터
        │
        ▼
Full Dataset Index 구축
        │
        ▼
Retrieval 평가 및 분석
```

---

## 🧩 Common Data Schema

온통청년 Open API에서 수집한 데이터를 RAG 파이프라인에서 일관되게 처리할 수 있도록 공통 데이터 형식으로 변환합니다.

```json
{
  "doc_id": "unique_policy_id",
  "title": "policy_title",
  "content": "policy_content",
  "category": "policy_category",
  "source": "youthcenter",
  "metadata": {}
}
```

평가 데이터는 검색 결과와 Ground Truth를 연결할 수 있도록 별도로 관리합니다.

```json
{
  "query_id": "query_001",
  "query": "청년이 받을 수 있는 주거 지원 정책은?",
  "ground_truth_chunk_ids": [
    "chunk_001"
  ]
}
```

---

## 📊 Evaluation

Baseline 단계에서는 우선 **Retriever의 검색 성능**을 중심으로 평가합니다.

| Metric    | Description                                     |
| --------- | ----------------------------------------------- |
| **Hit@k** | Top-k 검색 결과 안에 Ground Truth 청크가 하나 이상 포함되었는지 평가 |
| **MRR**   | 최초 Ground Truth 청크가 검색 결과에서 얼마나 높은 순위에 등장하는지 평가 |

### Hit@k

```text
Top-1 → 정답 없음
Top-2 → 정답 없음
Top-3 → 정답 청크 등장

Hit@1 = 0
Hit@3 = 1
```

### MRR

첫 번째 정답 청크의 순위를 이용합니다.

```text
정답 청크 순위 = 3

Reciprocal Rank = 1 / 3
                = 0.333
```

전체 질문의 Reciprocal Rank 평균을 계산하여 MRR을 구합니다.

초기에는 **Hit@k와 MRR**을 핵심 지표로 사용하고 프로젝트 진행에 따라 필요한 평가 지표를 추가합니다.

---

## 📁 Project Structure

```text
.
├── configs/              # RAG 및 실험 설정
│
├── data/
│   ├── raw/              # 온통청년 API 원본 데이터
│   ├── interim/          # 정제 및 중간 처리 데이터
│   └── processed/        # 최종 전처리 데이터
│
├── src/
│   ├── collection/       # Open API 데이터 수집
│   ├── preprocessing/    # 데이터 정제 및 스키마 변환
│   ├── sampling/         # Baseline 데이터 표본 추출
│   ├── chunking/         # 문서 Chunking
│   ├── embedding/        # Embedding 생성
│   ├── indexing/         # ChromaDB Index 구축
│   ├── retrieval/        # Top-k Retrieval
│   └── evaluation/       # Retrieval 성능 평가
│
├── experiments/          # 실험 설정 및 결과
│
├── indexes/              # Vector DB / Index 데이터
│
├── tests/
│   ├── fixtures/
│   ├── test_schema_validation.py
│   ├── test_qrels_mapping.py
│   ├── test_retrieval_metrics.py
│   └── test_pipeline.py
│
├── learning/             # 초기 RAG 학습 및 구현 기록
│
├── requirements.txt
└── README.md
```

---

## 🧪 Experiment Strategy

Baseline Dataset을 이용하여 다음 설정에 따른 Retrieval 성능 변화를 비교합니다.

```text
Chunk Size
Chunk Overlap
Embedding Model
Embedding Normalization
Top-k
Distance Metric
```

각 실험에서는 **설정값과 평가 결과를 함께 기록**하여 동일한 실험을 다시 실행하고 비교할 수 있도록 구성합니다.

Embedding Model, Embedding Dimension, Chunking 방식, Dataset Version 등 Index 자체에 영향을 주는 설정이 변경되는 경우 별도의 Index를 생성하여 실험 환경을 분리합니다.

---

## 🛠️ Tech Stack

### RAG / AI

* Python
* Sentence Transformers
* ChromaDB

### Evaluation

* Hit@k
* MRR

### Data

* 온통청년 Open API
* JSON / JSONL

### Collaboration

* Git
* GitHub
* Notion

---

## 🚀 Development Roadmap

```text
RAG 개념 학습 및 개별 구현
        ↓
MVP 목적 및 구조 정의
        ↓
공통 데이터 스키마 설계
        ↓
온통청년 Open API 데이터 수집
        ↓
데이터 정제 및 전처리
        ↓
카테고리별 Baseline 데이터 추출
        ↓
Baseline RAG 구축
        ↓
Ground Truth / 평가 데이터 구축
        ↓
Retrieval 평가 파이프라인 구축
        ↓
Hit@k / MRR Baseline 측정
        ↓
Chunking / Embedding / Retrieval 실험
        ↓
Baseline 설정 확정
        ↓
온통청년 전체 데이터 적용
        ↓
Full Dataset Index 구축
        ↓
Retrieval 성능 평가 및 분석
        ↓
RAG 시스템 개선
```

---

## 🔬 What We Focus On

본 프로젝트는 단순히 **RAG를 이용한 챗봇을 구현하는 것**보다 검색 시스템 자체를 이해하고 개선하는 과정에 중점을 둡니다.

주요 실험 질문은 다음과 같습니다.

* 정책 문서를 어떤 구조로 전처리해야 검색에 유리한가?
* Chunk Size와 Overlap에 따라 검색 성능은 어떻게 달라지는가?
* Embedding Model에 따라 정책 검색 성능은 어떻게 달라지는가?
* Top-k 값에 따라 Ground Truth 검색률은 어떻게 달라지는가?
* 검색 성능을 정량적으로 어떻게 평가할 수 있는가?
* 표본 데이터에서 결정한 설정이 전체 정책 데이터에서도 유효한가?
* 데이터 규모가 증가하면 Retrieval 성능은 어떻게 변화하는가?

이를 통해 RAG를 단순히 사용하는 것을 넘어 **데이터 → 검색 → 평가 → 실험 → 개선으로 이어지는 전체 Retrieval 파이프라인을 직접 설계하고 검증하는 것**을 목표로 합니다.

---

## 👥 Team

본 프로젝트는 팀 단위로 진행하며 데이터 수집 및 전처리, RAG 파이프라인 구축, Retrieval 평가, 실험 및 분석 등의 역할을 분담하여 진행합니다.

---

## 📌 Current Status

**현재 단계**

`온통청년 Open API 데이터 수집 및 Baseline Dataset 구축`

온통청년 Open API를 통해 전체 청년정책 데이터를 수집하고 있으며, 수집된 데이터를 공통 데이터 형식으로 변환한 뒤 카테고리별 표본 추출을 통해 Baseline Dataset을 구축할 예정입니다.

이후 해당 데이터셋을 기반으로 Baseline RAG와 Retrieval 평가 파이프라인을 구축합니다.
