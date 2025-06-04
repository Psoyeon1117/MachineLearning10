# 키워드 추출 프로젝트

문서의 핵심 내용을 나타내는 키워드를 추출하는 다양한 모델들을 비교하고 평가합니다.

## 구현된 모델

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**

   - 통계적 방법을 사용한 키워드 추출
   - 단어의 빈도와 역문서 빈도를 고려

2. **YAKE (Yet Another Keyword Extractor)**

   - 비지도 학습 기반 키워드 추출
   - 단어의 위치, 빈도, 관계성을 고려

3. **RAKE (Rapid Automatic Keyword Extraction)**

   - 구문 분석 기반 키워드 추출
   - 단어 동시 출현 그래프를 활용

4. **KeyBERT**
   - BERT 기반 의미론적 키워드 추출
   - 문서와 키워드 간의 의미적 유사성을 고려

## 파일 구조

```
project/
├── clean_data.csv              # 입력 데이터 (Document, Class 컬럼)
├── requirements.txt            # Python 패키지 의존성
├── setup.sh                   # 설치 및 설정 스크립트
├── run_test.sh                # 테스트 실행 스크립트
├── data_loader.py             # 데이터 로딩 유틸리티
├── tfidf_extractor.py         # TF-IDF 키워드 추출기
├── yake_extractor.py          # YAKE 키워드 추출기
├── rake_extractor.py          # RAKE 키워드 추출기
├── keybert_extractor.py       # KeyBERT 키워드 추출기
├── keyword_test.py            # 메인 테스트 스크립트
├── model_comparison.py        # 모델 성능 비교 스크립트
└── README.md                  # 이 파일
```

## 설치 및 설정

### 1. 의존성 설치

```bash
# 자동 설정 (권장)
chmod +x setup.sh
./setup.sh

# 또는 수동 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사용법

### 1. 기본 테스트

```bash
# 랜덤 문서 10개 테스트
./run_test.sh random

# 특정 문서 테스트 (예: 1000번째 문서)
./run_test.sh doc_1000
```

### 2. 모델 성능 비교

```bash
# 전체 모델 성능 비교
python3 model_comparison.py
```

### 3. 개별 모델 테스트

```bash
# 각 모델 개별 실행
python3 tfidf_extractor.py
python3 yake_extractor.py
python3 rake_extractor.py
python3 keybert_extractor.py
```

## 출력 예시

### 랜덤 테스트 결과

```
================================================================================
랜덤 문서 10개 테스트
================================================================================

[문서 1/10] 인덱스: 1500
내용 (처음 100자): Machine learning algorithms are designed to learn patterns from data...
--------------------------------------------------
TF-IDF  : ['machine learning', 'algorithms', 'patterns', 'data', 'neural'] (0.045초)
YAKE    : ['machine learning', 'deep learning', 'neural networks', 'data patterns', 'algorithms'] (0.032초)
RAKE    : ['machine learning algorithms', 'neural networks', 'data patterns', 'deep learning', 'artificial intelligence'] (0.028초)
KeyBERT : ['machine learning', 'algorithms', 'neural networks', 'data science', 'artificial intelligence'] (0.156초)
```

### 성능 비교 결과

```
================================================================================
모델 성능 비교 리포트
================================================================================
모델       평균시간    중간시간    성공률    키워드/문서    키워드길이
--------------------------------------------------------------------------------
TF-IDF     0.045      0.042      98.5%     4.8          12.3
YAKE       0.032      0.030      97.2%     4.9          14.1
RAKE       0.028      0.026      96.8%     4.7          16.8
KeyBERT    0.156      0.152      99.1%     5.0          13.6

================================================================================
성능 순위
================================================================================

속도 순위 (빠른 순):
  1. RAKE: 0.028초
  2. YAKE: 0.032초
  3. TF-IDF: 0.045초
  4. KeyBERT: 0.156초

성공률 순위 (높은 순):
  1. KeyBERT: 99.1%
  2. TF-IDF: 98.5%
  3. YAKE: 97.2%
  4. RAKE: 96.8%
```

## 모델별 특징

### TF-IDF

- **장점**: 빠른 처리 속도, 안정적 성능
- **단점**: 의미론적 관계 고려 부족
- **적합한 경우**: 대용량 텍스트 처리, 실시간 처리

### YAKE

- **장점**: 비지도 학습, 다국어 지원
- **단점**: 매개변수 조정 필요
- **적합한 경우**: 레이블 데이터 없는 경우

### RAKE

- **장점**: 매우 빠른 처리, 구문 정보 활용
- **단점**: 짧은 텍스트에서 성능 저하
- **적합한 경우**: 긴 문서, 학술 논문

### KeyBERT

- **장점**: 의미론적 유사성 고려, 높은 품질
- **단점**: 느린 처리 속도, 높은 메모리 사용
- **적합한 경우**: 품질이 중요한 경우, 배치 처리
