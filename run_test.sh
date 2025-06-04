#!/bin/bash

# 키워드 추출 모델 테스트 스크립트
# 사용법: ./run_test.sh [random|doc_INDEX]

# Python 가상환경 활성화 (필요한 경우)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "가상환경 활성화됨"
fi

# Python 스크립트 실행
echo "키워드 추출 테스트 시작..."
python3 keyword_test.py "$@"

echo "테스트 완료"
