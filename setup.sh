#!/bin/bash

# 키워드 추출 프로젝트 환경 설정 스크립트

echo "🚀 키워드 추출 프로젝트 환경 설정 시작"

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv가 설치되어 있지 않습니다."
    echo "💡 uv 설치 방법: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv 발견됨"

# Python 버전 확인 및 설치
echo "🐍 Python 환경 설정 중..."
uv python install 3.10.12

# 가상환경 생성 및 활성화
echo "📦 가상환경 생성 중..."
uv venv

# 의존성 설치
echo "📚 의존성 설치 중..."
uv pip install -e .

# 개발 의존성 설치 (선택사항)
read -p "🔧 개발 의존성도 설치하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install -e ".[dev]"
fi

# NLTK 데이터 다운로드
echo "📊 NLTK 데이터 다운로드 중..."
uv run python -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print('✅ NLTK 데이터 다운로드 완료')
except:
    print('⚠️ NLTK 데이터 다운로드 실패 (인터넷 연결 확인)')
"

echo "🎉 환경 설정 완료!"
echo ""
echo "📋 다음 단계:"
echo "1. 가상환경 활성화: source .venv/bin/activate"
echo "2. 테스트 실행: uv run python main.py help"
echo "3. 벤치마크 실행: uv run python main.py benchmark 5"
