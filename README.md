# 🔍 Multi-Model Keyword Extraction System

문서 내용을 바탕으로 핵심 키워드를 추출하는 다중 모델 시스템입니다. TF-IDF부터 DistilBERT까지 다양한 접근 방식을 비교 분석할 수 있습니다.

## 📋 프로젝트 개요

### 목표

- 문서의 내용을 바탕으로 핵심 키워드 3~5개 추출
- 다양한 모델의 성능 비교 및 분석
- 사전 학습된 모델을 활용한 inference 중심 접근

### 모델 구성

1. **Baseline**: TF-IDF + Logistic Regression / SVM
2. **Mid-level**: BiLSTM / GRU
3. **Advanced**: DistilBERT / KLUE BERT

## 🚀 빠른 시작

### 1. 환경 설정

#### uv 설치 (아직 설치하지 않은 경우)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 프로젝트 환경 설정

```bash
# 자동 설정 스크립트 실행
chmod +x setup.sh
./setup.sh

# 또는 수동 설정
uv venv
uv pip install -e .
```

### 2. 테스트 실행 권한 부여

```bash
chmod +x run_test.sh
```

### 3. 빠른 테스트

```bash
# 랜덤 문서로 빠른 테스트
./run_test.sh quick

# 또는 직접 실행
uv run python main.py random
```

## 📊 사용 방법

### 기본 명령어

```bash
# 도움말 확인
uv run python main.py help

# 벤치마크 테스트 (5개 샘플)
uv run python main.py benchmark 5

# 특정 문서 테스트
uv run python main.py test 100

# 랜덤 문서 테스트
uv run python main.py random
```

### 편의 스크립트 사용

```bash
# 빠른 테스트
./run_test.sh quick

# 벤치마크 테스트 (10개 샘플)
./run_test.sh benchmark 10

# 특정 문서 테스트
./run_test.sh test 50

# 개별 모델 테스트
./run_test.sh baseline   # Baseline 모델만
./run_test.sh midlevel   # Mid-level 모델만
./run_test.sh advanced   # Advanced 모델만
```
