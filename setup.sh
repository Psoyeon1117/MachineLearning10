#!/bin/bash

echo "키워드 추출 프로젝트 설정 시작..."

# Python 가상환경 생성 (이미 있으면 스킵)
if [ ! -d ".venv" ]; then
    echo "Python 가상환경 생성 중..."
    python3 -m venv .venv
fi

# 가상환경 활성화
echo "가상환경 활성화..."
source .venv/bin/activate

# 패키지 설치
echo "필요한 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# NLTK 데이터 다운로드
echo "NLTK 데이터 다운로드 중..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK 데이터 다운로드 완료')
except Exception as e:
    print(f'NLTK 다운로드 중 오류: {e}')
"

# 실행 권한 설정
chmod +x run_test.sh

echo "설정 완료!"
echo ""
echo "사용법:"
echo "  ./run_test.sh random          # 랜덤 문서 10개 테스트"
echo "  ./run_test.sh doc_1000        # 특정 문서 테스트"
echo "  python3 model_comparison.py   # 모델 성능 비교"
echo ""
echo "각 모델 개별 테스트:"
echo "  python3 tfidf_extractor.py"
echo "  python3 yake_extractor.py"
echo "  python3 rake_extractor.py"
echo "  python3 keybert_extractor.py"
